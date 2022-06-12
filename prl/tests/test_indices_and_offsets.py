"""Rolling indices between frontend, and backend needs care.
The prl_environment.steinberger.PokerRL-environment,
 has the observation and seat indices relative to the BTN.
The vectorized observation is always relative to the current player.
The frontend needs the seat indices relative to the HERO seat.

Additionally, when a players stack is 0 at the beginning of a round, he or she is eliminated.
His/her seat is kept in the frontend, and thus seat indices and their order does not change in the frontend.
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
On end of a round -that is signalled by the api via a `done`-flag response-
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

from prl.api.calls.environment.utils import get_player_cards
from prl.api.model.environment_state import PlayerInfo, Players

MAX_PLAYERS = 6


def update_button_seat_frontend(stacks: list, old_btn_seat: int):
    # old_btn_seat = 2
    # stacks = [0, 140, 800, 0, 0, 200]
    # new button seat should be 5 because 3,4 are eliminated
    rolled_stack_values = np.roll(stacks, -old_btn_seat)
    # rolled_stack_values = [800   0   0 200   0 140]
    for i, s in enumerate(rolled_stack_values):
        if i == 0: continue  # exclude old button
        if s > 0:
            # translate index i from rolled to unrolled stack list
            return (i + old_btn_seat) % MAX_PLAYERS
    raise ValueError('Not enough players with positive stacks to determine next button.')


def get_indices_map(stacks: list, new_btn_seat_frontend: int):
    seat_ids_remaining_frontend = [i for i, s in enumerate(stacks) if s > 0]  # [1, 2, 5]
    roll_by = -seat_ids_remaining_frontend.index(new_btn_seat_frontend)
    rolled_seat_ids = np.roll(seat_ids_remaining_frontend, roll_by)  # [5, 1, 2]
    # mapped_indices = dict(list(zip(seat_ids_remaining_frontend, rolled_seat_ids)))
    return dict([(pid_backend, seat_frontend) for pid_backend, seat_frontend in enumerate(rolled_seat_ids)])


def test_translate_starting_stacks_for_backend():
    # gotten from body.stack_sizes.dict()
    starting_stacks = {'stack_p0': 0,
                       'stack_p1': 140,
                       'stack_p2': 800,  # BTN
                       'stack_p3': 0,
                       'stack_p4': 0,
                       'stack_p5': 200}

    # 1. determine new frontend button seat
    # 2. compute map that translates backend indices back to frontend indices after stepping the environment
    # 3. make starting_stack_list for prl_environment.steinberger.PokerRL-environment

    # 1.
    btn_seat_frontend = 2
    new_btn_seat_frontend = update_button_seat_frontend(stacks=list(starting_stacks.values()),
                                                        old_btn_seat=btn_seat_frontend)
    # 2.
    mapped_indices = get_indices_map(stacks=list(starting_stacks.values()),
                                     new_btn_seat_frontend=new_btn_seat_frontend)
    # 3.
    rolled_stack_values = np.roll(list(starting_stacks.values()), -new_btn_seat_frontend)  # [200   0 140 800   0   0]
    seat_ids_with_pos_stacks = np.where(rolled_stack_values != 0)
    trimmed_rolled_stack_values = rolled_stack_values[seat_ids_with_pos_stacks]  # [200 140 800]
    assert np.array_equal(trimmed_rolled_stack_values, [200, 140, 800])
    assert mapped_indices == {0: 5,
                              1: 1,
                              2: 2}


def not_a_test_get_player_stats(obs, obs_keys, offset, mapped_indices: dict):

    observation_slices_per_player = []
    obs_keys = [re.sub(re.compile(r'p\d'), 'p', s) for s in obs_keys]
    for i in range(MAX_PLAYERS):
        start_idx = obs_keys.index(f'stack_p{i}')
        end_idx = obs_keys.index(f'side_pot_rank_p{i}_is_5') + 1
        observation_slices_per_player.append(slice(start_idx, end_idx))

    player_info = {}
    for pid, frontend_seat in mapped_indices.items():

        hand = get_player_cards(idx_start=obs_keys.index(f"{pid}th_player_card_0_rank_0"),
                                idx_end=obs_keys.index(f"{pid}th_player_card_1_suit_3") + 1,
                                obs=obs)
        p_info = list(zip(obs_keys, obs))[observation_slices_per_player[pid]]

        player_info[f'p{frontend_seat}'] = PlayerInfo(**{'pid': frontend_seat, **dict(p_info), **dict(hand)})

    p_info_rolled = np.roll(list(player_info.values()), offset, axis=0)
    p_info_rolled = dict(list(zip(player_info.keys(), p_info_rolled)))
    response_players = Players(**p_info_rolled)


    if __name__ == '__main__':
        test_translate_starting_stacks_for_backend()
