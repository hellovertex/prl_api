"""Rolling indices between frontend, and backend needs care.
The prl_environment.steinberger.PokerRL-environment,
 has the observation and seat indices relative to the BTN.
The vectorized observation is always relative to the current player.
The frontend needs the seat indices relative to the HERO seat.

Additionally, when a players stack is 0 at the beginning of a round, he or she is eliminated.
His seat is kept in the frontend, and thus seat indices and their order does not change in the frontend.
However, the prl_environment.steinberger.PokerRL-environment, needs to be reset
with a decreased number of players, because it does not work,
if it is reset with a person that has a stack size of 0.

Furthermore, we need to keep track of the BTN position. Lets look at an example

Round starts with 6 players and stacks. The button is randomly determined at
the very first reset, and gets seat_id=2. After that the BTN goes to the next
 player on the left.
These are the starting stacks of round=1 as seen by the frontend.
{
    200,  # HERO
    140,
    200,  # BTN
    200,
    200
    200
}
During the game, 3 players go all in and we get final stacks
{
    0,    # HERO
    140,
    800,  # BTN
    0,
    0
    200
}

Now the button must be propagated to seat_id=5.
On end of a round that is signalled by the api via a `done`-flag response,
the frontend sends a POST request to the /reset endpoint of the API using the final stack sizes.

It is now the API`s job -upon receiving the reset-request, to translate
{
    0,    # HERO
    140,
    800,  # BTN
    0,
    0
    200
}

to
{
    200,  # BTN propagated to the left
    140,
    800,
} for the backend to reset the environment
 - using the new BTN position
 - and without eliminated players,
and translate it back for the frontend to
{
    0,    # HERO
    140,
    800,
    0,
    0
    200  # BTN
}
"""

import re

import numpy as np
from prl.environment.steinberger.PokerRL.game import Poker

from prl.api.model.environment_state import PlayerInfo, Card, Board, Table, Players

MAX_PLAYERS = 6
RANK_DICT = {
    Poker.CARD_NOT_DEALT_TOKEN_1D: "",
    0: "2",
    1: "3",
    2: "4",
    3: "5",
    4: "6",
    5: "7",
    6: "8",
    7: "9",
    8: "T",
    9: "J",
    10: "Q",
    11: "K",
    12: "A"
}
SUIT_DICT = {
    Poker.CARD_NOT_DEALT_TOKEN_1D: "",
    0: "h",
    1: "d",
    2: "s",
    3: "c"
}


def get_indices_map(stacks: list, new_btn_seat_frontend: int):
    """ Gets a list of stacks. Rolls all non-zero stacks relative to button,
    and returns a map of the indices mapping from button-view back to the initial view"""
    seat_ids_remaining_frontend = [i for i, s in enumerate(stacks) if s > 0]  # [1, 2, 5]
    roll_by = -seat_ids_remaining_frontend.index(new_btn_seat_frontend)
    rolled_seat_ids = np.roll(seat_ids_remaining_frontend, roll_by)  # [5, 1, 2]
    # mapped_indices = dict(list(zip(seat_ids_remaining_frontend, rolled_seat_ids)))
    return dict([(pid_backend, seat_frontend) for pid_backend, seat_frontend in enumerate(rolled_seat_ids)])


def update_button_seat_frontend(stacks: list, old_btn_seat: int):
    """Rolls stacks relative to button view and picks the first non-zero stack
     after button to determine its position as the new button position."""
    # old_btn_seat = 2
    # stacks = [0, 140, 800, 0, 0, 200]
    # new button seat should be 5 because 3,4 are eliminated
    rolled_stack_values = np.roll(stacks, -old_btn_seat)
    rolled_stack_values[rolled_stack_values == None] = 0

    # rolled_stack_values = [800   0   0 200   0 140]
    for i, s in enumerate(rolled_stack_values):
        if i == 0: continue  # exclude old button
        if s > 0:
            # translate index i from rolled to unrolled stack list
            return (i + old_btn_seat) % MAX_PLAYERS
    raise ValueError('Not enough players with positive stacks to determine next button.')


def maybe_replace_leading_digit(val):
    val = val.replace('0th', 'first')
    val = val.replace('1th', 'second')
    val = val.replace('2th', 'third')
    val = val.replace('3th', 'fourth')
    val = val.replace('4th', 'fifth')
    return val.replace('5th', 'sixth')


def get_player_cards(idx_start, idx_end, obs, n_suits=4, n_ranks=13):
    cur_idx = idx_start
    cards = {}
    end_idx = 0
    for i in range(2):
        suit = -127
        rank = -127
        end_idx = cur_idx + n_suits + n_ranks
        bits = obs[cur_idx:end_idx]
        # print(f'obs[cur_idx:end_idx] = {obs[cur_idx:end_idx]}')
        if sum(bits) > 0:
            idx = np.where(bits == 1)[0]
            rank, suit = idx[0], idx[1] - n_ranks

        cards[f'c{i}'] = Card(**{'name': RANK_DICT[rank] + SUIT_DICT[suit],
                                 'suit': suit,
                                 'rank': rank,
                                 'index': i})
        cur_idx = end_idx
    assert end_idx == idx_end
    return cards


def get_player_stats(obs, obs_keys, offset, mapped_indices: dict, normalization):
    observation_slices_per_player = []

    for i in range(MAX_PLAYERS):
        start_idx = obs_keys.index(f'stack_p{i}')
        end_idx = obs_keys.index(f'side_pot_rank_p{i}_is_5') + 1
        observation_slices_per_player.append(slice(start_idx, end_idx))

    player_info = {}
    obs_keys = [re.sub(re.compile(r'p\d'), 'p', s) for s in obs_keys]
    for pid, frontend_seat in mapped_indices.items():
        hand = get_player_cards(idx_start=obs_keys.index(f"{pid}th_player_card_0_rank_0"),
                                idx_end=obs_keys.index(f"{pid}th_player_card_1_suit_3") + 1,
                                obs=obs)
        p_info = dict(list(zip(obs_keys, obs))[observation_slices_per_player[pid]])
        p_info['stack_p'] = round(p_info['stack_p'] * normalization)
        p_info['curr_bet_p'] = round(p_info['curr_bet_p'] * normalization)
        player_info[f'p{frontend_seat}'] = PlayerInfo(**{'pid': frontend_seat, **p_info, **dict(hand)})

    p_info_rolled = np.roll(list(player_info.values()), offset, axis=0)
    p_info_rolled = dict(list(zip(player_info.keys(), p_info_rolled)))
    response_players = Players(**p_info_rolled)
    return response_players


def get_board_cards(idx_board_start, idx_board_end, obs, n_suits=4, n_ranks=13):
    cur_idx = idx_board_start
    cards = {}
    end_idx = 0
    for i in range(5):
        suit = -127
        rank = -127
        end_idx = cur_idx + n_suits + n_ranks
        bits = obs[cur_idx:end_idx]
        if sum(bits) > 0:
            idx = np.where(bits == 1)[0]
            rank, suit = idx[0], idx[1] - n_ranks

        cards[f'b{i}'] = Card(**{'name': RANK_DICT[rank] + SUIT_DICT[suit],
                                 'suit': suit,
                                 'rank': rank,
                                 'index': i})
        cur_idx = end_idx
    # print(f'idx_board_end = {idx_board_end}')
    # print(f'end_idx = {end_idx}')
    assert idx_board_end == end_idx
    return Board(**cards)


def get_table_info(obs_keys, obs, observer_offset, normalization, map_indices):
    """Observer offset is necessary to compensate for the fact,
    that the vectorized observation is not relative to hero or button, but it
    is relative to the next acting player.

    Since map_indices computes all indices relative to button, we must roll
    these a last time by the seat_id of the next player.

    For example [0,1,2,3,4,5] frontend seat ids with a button at seat_id=2 becomes
    [2,3,4,5,0,1] relative to the button. If the acting player is CO, he has a seat_id=1
    given that button is at seat_id=2. Hence the observation relative to the acting playre
    is relative to seat_id=1 and becomes [1,2,3,4,5,0].

    Rolling [2,3,4,5,0,1] that we obtained from applying the indices_map by
    The CO seat_id, i.e. 1, we obtain exactly the order of [1,2,3,4,5,0] that is
    used in the observation vector.
    """

    side_pots: np.ndarray = np.zeros(MAX_PLAYERS)
    for pid, seat in map_indices.items():
        side_pots[seat] = obs[obs_keys.index(f'side_pot_{pid}')]
    sp_keys = ['side_pot_0', 'side_pot_1', 'side_pot_2', 'side_pot_3', 'side_pot_4', 'side_pot_5']

    side_pots = np.roll(side_pots, observer_offset)

    table = {'ante': round(obs[obs_keys.index('ante')] * normalization),
             'small_blind': round(obs[obs_keys.index('small_blind')] * normalization),
             'big_blind': round(obs[obs_keys.index('big_blind')] * normalization),
             'min_raise': round(obs[obs_keys.index('min_raise')] * normalization),
             'pot_amt': round(obs[obs_keys.index('pot_amt')] * normalization),
             'total_to_call': round(obs[obs_keys.index('total_to_call')] * normalization),
             'round_preflop': obs[obs_keys.index('round_preflop')],
             'round_flop': obs[obs_keys.index('round_flop')],
             'round_turn': obs[obs_keys.index('round_turn')],
             'round_river': obs[obs_keys.index('round_river')],
             # side pots 0 to 5
             **dict(list(zip(sp_keys, side_pots)))
             }
    # table_kwargs = list(zip(obs_keys, obs))[0:obs_keys.index('side_pot_5') + 1]
    # return Table(**dict(table_kwargs))
    return Table(**table)


def get_stacks(player_info):
    stacks = {}
    for pid, pinfo in player_info.dict().items():
        try:
            stacks[pid] = int(pinfo['stack_p'])
        except TypeError:
            stacks[pid] = 0

    return stacks
