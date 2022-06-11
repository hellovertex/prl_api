from __future__ import annotations

import numpy as np
from fastapi import APIRouter
from prl.environment.Wrappers.prl_wrappers import AgentObservationType
from prl.environment.steinberger.PokerRL import NoLimitHoldem
from starlette.requests import Request

from prl.api.calls.environment.utils import get_table_info, get_board_cards, get_player_stats, get_rolled_stack_sizes
from prl.api.model.environment_reset import EnvironmentResetRequestBody
from prl.api.model.environment_state import EnvironmentState, Info

router = APIRouter()
abbrevs = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth']
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


def move_button_to_next_available_frontend_seat(request, env_id):
    stacks = request.body().stack_sizes.dict().values()
    old_btn_seat = request.app.backend.metadata[env_id]['button_index']
    new_btn_seat = update_button_seat_frontend(stacks, old_btn_seat)
    request.app.backend.metadata[env_id]['button_index'] = new_btn_seat


def roll_starting_stacks_relative_to_button(request, body, env_id, n_players):
    """Roll stack sizes back such that button is at position 0.
    This is required because the frontend always sends the stacks relative to the hero position."""
    button_index = request.app.backend.metadata[env_id]['button_index']
    stack_sizes_rolled = None
    if body.stack_sizes is None:
        # 1. fall back to default stack size if no stacks were provided in request
        default_stack = request.app.backend.active_ens[env_id].env.DEFAULT_STACK_SIZE
        stack_sizes_rolled = [default_stack for _ in range(n_players)]
    else:
        # 2. set custom stack sizes provided in the request body
        if not request.app.backend.metadata[env_id]['initial_state']:
            stack_sizes_rolled = np.roll(list(body.stack_sizes.dict().values()), -button_index, axis=0)
            try:
                stack_sizes_rolled = [stack.item() for stack in stack_sizes_rolled]
            except Exception:
                pass
    return stack_sizes_rolled


def get_indices_map(stacks: list, new_btn_seat_frontend: int):
    # todo remove after debugging new_btn_seat_frontend = 5
    seat_ids_remaining_frontend = [i for i, s in enumerate(stacks) if s > 0]  # [1, 2, 5]
    roll_by = -seat_ids_remaining_frontend.index(new_btn_seat_frontend)
    rolled_seat_ids = np.roll(seat_ids_remaining_frontend, roll_by)  # [5, 1, 2]
    # mapped_indices = dict(list(zip(seat_ids_remaining_frontend, rolled_seat_ids)))
    return dict([(pid_backend, seat_frontend) for pid_backend, seat_frontend in enumerate(rolled_seat_ids)])


def translate_frontend_stack_sizes_to_environment_starting_stacks():
    pass


def assign_button_to_random_frontend_seat(request, body):
    # Randomly determine first button seat position in frontend
    stacks = np.array(list(body.stack_sizes.dict().values()))
    new_btn_seat_frontend = np.random.choice(np.where(stacks > 0)[0])
    request.app.backend.metadata[body.env_id]['initial_state'] = False
    request.app.backend.metadata[body.env_id]['button_index'] = new_btn_seat_frontend


@router.post("/environment/{env_id}/reset/",
             response_model=EnvironmentState,
             operation_id="reset_environment")
async def reset_environment(body: EnvironmentResetRequestBody, request: Request):
    env_id = body.env_id

    # if no starting stacks are provided we can safely get number of players from environment configuration
    # otherwise, starting stacks provided by the client indicate a maybe reduced number of players
    # DEFAULTS
    n_players = request.app.backend.active_ens[env_id].env.N_SEATS
    default_stack = request.app.backend.active_ens[env_id].env.DEFAULT_STACK_SIZE
    stack_sizes_rolled = [default_stack for _ in range(n_players)]
    mapped_indices = dict(list(zip([i for i in range(n_players)], [i for i in range(n_players)])))

    # Move Button to next available frontend seat
    if request.app.backend.metadata[env_id]['initial_state']:
        assign_button_to_random_frontend_seat(request, body)
    else:
        move_button_to_next_available_frontend_seat(request, env_id)
    new_btn_seat_frontend = request.app.backend.metadata[env_id]['button_index']

    # If request body contains stack sizes, remove zeros and roll them relative to BUTTON
    if body.stack_sizes:
        stacks = list(body.stack_sizes.dict().values())
        mapped_indices = get_indices_map(stacks=stacks, new_btn_seat_frontend=new_btn_seat_frontend)
        rolled_stack_values = np.roll(stacks, -new_btn_seat_frontend)  # [200   0 140 800   0   0]
        seat_ids_with_pos_stacks = np.where(rolled_stack_values != 0)
        stack_sizes_rolled = rolled_stack_values[seat_ids_with_pos_stacks]
        n_players = len(stack_sizes_rolled)
    stack_sizes_rolled = [round(s) for s in stack_sizes_rolled]

    # Set env_args such that rolled starting stacks are used
    args = NoLimitHoldem.ARGS_CLS(n_seats=n_players,
                                  starting_stack_sizes_list=stack_sizes_rolled,
                                  use_simplified_headsup_obs=False)
    request.app.backend.active_ens[env_id].overwrite_args(args, agent_observation_mode=AgentObservationType.SEER)
    obs_dict = request.app.backend.active_ens[env_id].obs_idx_dict
    obs_keys = [k for k in obs_dict.keys()]
    obs, _, _, _ = request.app.backend.active_ens[env_id].reset()

    # offset that moves observation from relativ to current seat to relative to hero offset
    # when we have the observation relative to hero offset, we can apply our indices map from above
    # to map to the seat ids in the frontend
    pid_next_to_act_backend = request.app.backend.active_ens[env_id].env.current_player.seat_id
    offset_current_player_to_hero = pid_next_to_act_backend
    normalization = request.app.backend.active_ens[env_id].normalization
    # table_info = get_table_info(obs_keys, obs, offset=offset, n_players=n_players, normalization=normalization)
    table_info = get_table_info(obs_keys, obs,
                                observer_offset=offset_current_player_to_hero,
                                n_players=n_players,
                                normalization=normalization,
                                map_indices=mapped_indices)

    board_cards = get_board_cards(idx_board_start=obs_keys.index('0th_board_card_rank_0'),
                                  idx_board_end=obs_keys.index('0th_player_card_0_rank_0'),
                                  obs=obs)

    player_info = get_player_stats(obs=obs,
                                   obs_keys=obs_keys,
                                   offset=offset_current_player_to_hero,
                                   mapped_indices=mapped_indices,
                                   normalization=normalization)

    stack_sizes_rolled = get_rolled_stack_sizes(request, body, n_players, new_btn_seat_frontend)
    result = {'env_id': env_id,
              'n_players': n_players,
              'stack_sizes': stack_sizes_rolled,
              'last_action': None,
              'table': table_info,
              'players': player_info,
              'board': board_cards,
              'button_index': new_btn_seat_frontend,
              'p_acts_next': mapped_indices[0] if n_players < 4 else mapped_indices[3],
              'done': False,
              'info': Info(**{'continue_round': True,
                              'draw_next_stage': False,
                              'rundown': False,
                              'deal_next_hand': False,
                              'payouts': None})
              }
    return EnvironmentState(**dict(result))
