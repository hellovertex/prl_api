from __future__ import annotations
from random import randint
from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel
from starlette.requests import Request
import numpy as np

from PokerRL import NoLimitHoldem
from src.calls.environment.utils import get_table_info, get_board_cards, get_player_stats, get_rolled_stack_sizes
from src.model.environment_state import EnvironmentState, Info, Players

router = APIRouter()
abbrevs = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth']


class Stacks(BaseModel):
    stack_p0: Optional[int]
    stack_p1: Optional[int]
    stack_p2: Optional[int]
    stack_p3: Optional[int]
    stack_p4: Optional[int]
    stack_p5: Optional[int]


class EnvironmentResetRequestBody(BaseModel):
    env_id: int
    stack_sizes: Optional[Stacks]

    class Config:
        schema_extra = {
            "env_id": {
                "example": 1,
                "description": "The environment unique id "
                               "used for requesting this specific environment."
            },
            "stack_sizes": {
                "example": 20000,
                "description": "The number of chips each player will get on resetting the environment."
                               "Note that the environment is reset on each hand dealt. This implies"
                               "that starting stacks can vary between players, e.g. in the middle of the game."
            }
        }


def set_button_index(request, env_id, n_players):
    if request.app.backend.metadata[env_id]['initial_state']:
        # 1. randomly determine first button holder
        button_index = randint(0, n_players - 1)
        request.app.backend.metadata[env_id]['initial_state'] = False
        request.app.backend.metadata[env_id]['button_index'] = button_index
    else:
        # 2. move button +1 to the left
        request.app.backend.metadata[env_id]['button_index'] += 1
        if request.app.backend.metadata[env_id]['button_index'] >= n_players:
            request.app.backend.metadata[env_id]['button_index'] = 0


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


@router.post("/environment/{env_id}/reset/",
             response_model=EnvironmentState,
             operation_id="reset_environment")
async def reset_environment(body: EnvironmentResetRequestBody, request: Request):
    env_id = body.env_id
    number_non_null_stacks = [stack for stack in body.stack_sizes.dict().values() if stack is not None]
    # if no starting stacks are provided we can safely get number of players from environment configuration
    # otherwise, starting stacks provided by the client indicate a maybe reduced number of players
    n_players = request.app.backend.active_ens[env_id].env.N_SEATS if (body.stack_sizes is None) else len(
        number_non_null_stacks)

    # 1. set button index
    set_button_index(request, env_id, n_players)

    # 2. roll stack sizes
    starting_stack_sizes_rolled = roll_starting_stacks_relative_to_button(request, body, env_id, n_players)

    # set env_args such that new starting stacks are used
    args = NoLimitHoldem.ARGS_CLS(n_seats=n_players,
                                  starting_stack_sizes_list=starting_stack_sizes_rolled,
                                  use_simplified_headsup_obs=False)
    request.app.backend.active_ens[env_id].overwrite_args(args)
    obs, _, _, _ = request.app.backend.active_ens[env_id].reset()

    obs_dict = request.app.backend.active_ens[env_id].obs_idx_dict
    obs_keys = [k for k in obs_dict.keys()]

    button_index = request.app.backend.metadata[env_id]['button_index']
    # offset relative to hero offset
    p_acts_next = request.app.backend.active_ens[env_id].env.current_player.seat_id
    offset = (p_acts_next + button_index) % n_players
    table_info = get_table_info(obs_keys, obs, offset=offset)

    idx_end_table = obs_keys.index('side_pot_5')
    board_cards = get_board_cards(idx_board_start=obs_keys.index('0th_board_card_rank_0'),
                                  idx_board_end=obs_keys.index('0th_player_card_0_rank_0'),
                                  obs=obs)
    player_info = get_player_stats(obs_keys, obs, start_idx=idx_end_table + 1, offset=offset, n_players=n_players)

    stack_sizes_rolled = get_rolled_stack_sizes(request, body, n_players, button_index)
    result = {'env_id': env_id,
              'n_players': n_players,
              'stack_sizes': stack_sizes_rolled,
              'last_action': None,
              'table': table_info,
              'players': player_info,
              'board': board_cards,
              'button_index': button_index,
              'p_acts_next':  (p_acts_next + button_index) % n_players,
              'done': False,
              'info': Info(**{'continue_round': True,
                              'draw_next_stage': False,
                              'rundown': False,
                              'deal_next_hand': False,
                              'payouts': None})
              }
    return EnvironmentState(**dict(result))
