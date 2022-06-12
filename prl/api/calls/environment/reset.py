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
from __future__ import annotations

import numpy as np
from fastapi import APIRouter
from prl.environment.Wrappers.prl_wrappers import AgentObservationType
from prl.environment.steinberger.PokerRL import NoLimitHoldem
from starlette.requests import Request

from prl.api.calls.environment.utils import get_table_info, get_board_cards, get_player_stats, get_stacks, \
    update_button_seat_frontend, get_indices_map
from prl.api.model.environment_reset import EnvironmentResetRequestBody
from prl.api.model.environment_state import EnvironmentState, Info

router = APIRouter()
abbrevs = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth']
MAX_PLAYERS = 6


def move_button_to_next_available_frontend_seat(request, body):
    stacks = list(body.stack_sizes.dict().values())
    old_btn_seat = request.app.backend.metadata[body.env_id]['button_index']
    if 'last_stack_sizes' in request.app.backend.metadata[body.env_id]:
        stacks = list(request.app.backend.metadata[body.env_id]['last_stack_sizes'].values())
    new_btn_seat = update_button_seat_frontend(stacks, old_btn_seat)
    request.app.backend.metadata[body.env_id]['button_index'] = new_btn_seat


def assign_button_to_random_frontend_seat(request, body):
    # todo this is called with possibly fucked up body.stack_sizes
    #  this needs to be called with last_stack_sizes if possible
    #  otherwise it will cause the env and vectorizer to throw random bugs
    # Randomly determine first button seat position in frontend
    stacks = np.array(list(body.stack_sizes.dict().values()))
    stacks[stacks == None] = 0
    available_pids = np.where(stacks > 0)[0]
    new_btn_seat_frontend = np.random.choice(available_pids)

    request.app.backend.metadata[body.env_id]['initial_state'] = False
    request.app.backend.metadata[body.env_id]['button_index'] = new_btn_seat_frontend
    n_players_alive = len(available_pids)
    # set sb and bb too for convenience
    tmp = np.roll(available_pids, -new_btn_seat_frontend)
    if n_players_alive <= 2:
        sb = tmp[0]
        bb = tmp[1]
    else:
        sb = tmp[1]
        bb = tmp[2]
    request.app.backend.metadata[body.env_id]['sb'] = sb
    request.app.backend.metadata[body.env_id]['bb'] = bb


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
        move_button_to_next_available_frontend_seat(request, body)
    new_btn_seat_frontend = request.app.backend.metadata[env_id]['button_index']

    # If request body contains stack sizes, remove zeros and roll them relative to BUTTON
    if body.stack_sizes:
        stacks = np.array(list(body.stack_sizes.dict().values()))
        stacks[stacks == None] = 0
        stacks = stacks.tolist()
        skip = True
        for s in stacks:
            if s != 0:
                skip = False
        # if stacks were empty, use last_stack_sizes
        if skip:
            stacks = list(request.app.backend.metadata[body.env_id]['last_stack_sizes'].values())
        mapped_indices = get_indices_map(stacks=stacks, new_btn_seat_frontend=new_btn_seat_frontend)
        rolled_stack_values = np.roll(stacks, -new_btn_seat_frontend)  # [200   0 140 800   0   0]
        seat_ids_with_pos_stacks = np.where(rolled_stack_values != 0)
        stack_sizes_rolled = rolled_stack_values[seat_ids_with_pos_stacks]
        n_players = len(stack_sizes_rolled)
    stack_sizes_rolled = [round(s) for s in stack_sizes_rolled]
    request.app.backend.metadata[env_id]['mapped_indices'] = mapped_indices

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
    table_info = get_table_info(obs_keys=obs_keys,
                                obs=obs,
                                observer_offset=offset_current_player_to_hero,
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

    stack_sizes = get_stacks(player_info)
    request.app.backend.metadata[body.env_id]['last_stack_sizes'] = stack_sizes
    result = {'env_id': env_id,
              'n_players': n_players,
              'stack_sizes': stack_sizes,
              'last_action': None,
              'table': table_info,
              'players': player_info,
              'board': board_cards,
              'button_index': new_btn_seat_frontend,
              'sb': mapped_indices[request.app.backend.metadata[env_id]['sb']],
              'bb': mapped_indices[request.app.backend.metadata[env_id]['bb']],
              'p_acts_next': mapped_indices[0] if n_players < 4 else mapped_indices[3],
              'done': False,
              'info': Info(**{'continue_round': True,
                              'draw_next_stage': False,
                              'rundown': False,
                              'deal_next_hand': False,
                              'payouts': None})
              }
    return EnvironmentState(**dict(result))
