"""Rolling seat indices between frontend, and backend needs care.
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
from __future__ import annotations

import numpy as np
from fastapi import APIRouter
from starlette.requests import Request

from calls.environment.utils import get_table_info, get_board_cards, get_player_stats, get_stacks, \
    update_button_seat_frontend, get_indices_map
from model.environment_reset import EnvironmentResetRequestBody
from model.environment_state import EnvironmentState, Info
from prl.environment.Wrappers.prl_wrappers import AgentObservationType
from prl.environment.steinberger.PokerRL import NoLimitHoldem

router = APIRouter()
abbrevs = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth']
MAX_PLAYERS = 6


def get_blinds(stacks: np.ndarray, button_index: int):
    """stacks are relative to hero position, how frontend uses them"""
    stacks[np.where(stacks == None)] = 0
    available_pids = np.where(stacks > 0)[0]
    n_players_alive = len(available_pids)
    tmp = np.roll(available_pids, -button_index)
    if n_players_alive <= 2:
        sb = tmp[0]
        bb = tmp[1]
    else:
        sb = tmp[1]
        bb = tmp[2]
    return sb, bb


def move_button_to_next_available_frontend_seat(env_id, request, stacks: list):
    """Move button position. Skip eliminated players."""
    old_btn_seat = request.app.backend.metadata[env_id]['button_index']
    new_btn_seat_frontend = update_button_seat_frontend(stacks, old_btn_seat)
    request.app.backend.metadata[env_id]['button_index'] = new_btn_seat_frontend


def assign_button_to_random_frontend_seat(env_id, request, stacks: list):
    """Randomly determine first button seat position in frontend."""
    # Randomly determine first button seat position in frontend
    stacks = np.array(stacks)
    stacks[stacks == None] = 0  # [200. None 140. 800. None None]
    available_pids = np.where(stacks > 0)[0]  # [200.   0. 140. 800.   0.   0.]
    new_btn_seat_frontend = np.random.choice(available_pids)  # pick from [0 2 3]

    request.app.backend.metadata[env_id]['button_index'] = new_btn_seat_frontend


def stack_sizes_valid(stacks: list):
    stacks = np.array(stacks)
    stacks[stacks == None] = 0
    # stacks = [s for s in stacks]
    valid = False
    for s in stacks:
        if s != 0:
            valid = True
    return valid


def try_get_stacks(request, body) -> list:
    """Try loading stack sizes from request body. If these are invalid,
    tries loading stack sizes from last played hand. If this fails,
    it returns default stack size for each player."""
    n_players = request.app.backend.active_ens[body.env_id].env.N_SEATS
    default_stack = request.app.backend.active_ens[body.env_id].env.DEFAULT_STACK_SIZE
    stacks = [default_stack for _ in range(n_players)]
    if body.stack_sizes:
        request_stacks = list(body.stack_sizes.dict().values())
        if stack_sizes_valid(request_stacks):
            stacks = request_stacks
        else:
            try:
                stacks = list(request.app.backend.metadata[body.env_id]['last_stack_sizes'].values())
            except KeyError:
                # no last round played
                pass
    return stacks


@router.post("/environment/{env_id}/reset/",
             response_model=EnvironmentState,
             operation_id="reset_environment")
async def reset_environment(body: EnvironmentResetRequestBody, request: Request):
    # DEFAULTS
    env_id = body.env_id

    # Parse stacks from body, if invalid, try loading stacks from last round, if fails, use default
    stacks = try_get_stacks(request, body)  # stacks relative to hero

    # 2. Move Button to next available frontend seat
    if request.app.backend.metadata[env_id]['initial_state']:
        assign_button_to_random_frontend_seat(env_id, request, stacks)  # stacks relative to hero
        # reset old stacks
        request.app.backend.metadata[body.env_id]['last_stack_sizes'] = list()
        request.app.backend.metadata[env_id]['initial_state'] = False
    else:
        move_button_to_next_available_frontend_seat(env_id, request, stacks)  # stacks relative to hero
    new_btn_seat_frontend = request.app.backend.metadata[env_id]['button_index']

    mapped_indices = get_indices_map(stacks=stacks, new_btn_seat_frontend=new_btn_seat_frontend)

    # 3. On un-rolled, un-trimmed stacks, apply transformation for backend
    # [None 200. None 140. 800. None]
    # stacks_sizes = {}
    # tmp = np.array(stacks)
    # tmp[np.where(tmp==None)] = 0
    # for i, s in enumerate(tmp):
    #     stacks_sizes[f'p{i}'] = round(s)
    rolled_stack_values = np.roll(stacks, -new_btn_seat_frontend)  # [200. None 140. 800. None None]
    rolled_stack_values[np.where(rolled_stack_values == None)] = 0  # [200.   0. 140. 800.   0.   0.]

    seat_ids_with_pos_stacks = np.where(rolled_stack_values != 0)  # [0 2 3]
    stack_sizes_rolled = rolled_stack_values[seat_ids_with_pos_stacks]  # [200. 140. 800.]
    n_players = len(stack_sizes_rolled)  # 3
    stack_sizes_rolled = [round(s) for s in stack_sizes_rolled]  # [200 140 800]
    request.app.backend.metadata[env_id]['mapped_indices'] = mapped_indices  # {0: 0, 1: 2, 2:3}

    # Set env_args such that rolled starting stacks are used
    args = NoLimitHoldem.ARGS_CLS(n_seats=n_players,
                                  starting_stack_sizes_list=stack_sizes_rolled,
                                  use_simplified_headsup_obs=False)
    request.app.backend.active_ens[env_id].overwrite_args(args,
                                                          agent_observation_mode=AgentObservationType.SEER,
                                                          n_players=n_players)
    obs_dict = request.app.backend.active_ens[env_id].obs_idx_dict
    obs_keys = [k for k in obs_dict.keys()]
    obs, _, _, _ = request.app.backend.active_ens[env_id].reset()
    request.app.backend.metadata[env_id]['last_obs'] = obs

    # offset that moves observation from relativ to current seat to relative to hero offset
    # when we have the observation relative to hero offset, we can apply our indices map from above
    # to map to the seat ids in the frontend
    pid_next_to_act_backend = request.app.backend.active_ens[env_id].env.current_player.seat_id
    offset_current_player_to_hero = pid_next_to_act_backend
    normalization = request.app.backend.active_ens[env_id].normalization
    # table_info = get_table_info(obs_keys, obs, offset=offset, n_players=n_players, normalization=normalization)
    table_info = get_table_info(obs_keys=obs_keys,
                                obs=obs,
                                observer_offset=offset_current_player_to_hero,
                                normalization=normalization,
                                map_indices=mapped_indices)

    # board_cards = get_board_cards(idx_board_start=obs_keys.index('0th_board_card_rank_0'),
    #                               idx_board_end=obs_keys.index('0th_player_card_0_rank_0'),
    #                               obs=obs)
    board_cards = get_board_cards(request.app.backend.active_ens[env_id].env.board)

    player_info = get_player_stats(obs=obs,
                                   obs_keys=obs_keys,
                                   offset=offset_current_player_to_hero,
                                   mapped_indices=mapped_indices,
                                   normalization=normalization)

    # small blind an big blind have been removed, need to add them back to stacks manually
    stack_sizes = get_stacks(player_info)
    request.app.backend.metadata[body.env_id]['last_stack_sizes'] = stack_sizes

    request.app.backend.metadata[env_id]['sb'] = mapped_indices[request.app.backend.active_ens[env_id].env.SB_POS]
    request.app.backend.metadata[env_id]['bb'] = mapped_indices[request.app.backend.active_ens[env_id].env.BB_POS]
    result = {'env_id': env_id,
              'n_players': n_players,
              'stack_sizes': stack_sizes,
              'last_action': None,
              'table': table_info,
              'players': player_info,
              'board': board_cards,
              'button_index': new_btn_seat_frontend,
              'sb': request.app.backend.metadata[env_id]['sb'],
              'bb': request.app.backend.metadata[env_id]['bb'],
              'p_acts_next': mapped_indices[0] if n_players < 4 else mapped_indices[3],
              'game_over': False,  # whole game
              'done': False,  # this hand
              'info': Info(**{'continue_round': True,
                              'draw_next_stage': False,
                              'rundown': False,
                              'deal_next_hand': False,
                              'payouts': None})
              }
    return EnvironmentState(**dict(result))
