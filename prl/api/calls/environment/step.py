from fastapi import APIRouter
from pydantic import BaseModel
from starlette.requests import Request
import numpy as np
from src.model.environment_state import EnvironmentState, EnvironmentState, LastAction, Info, Players
from .utils import get_table_info, get_board_cards, get_player_stats, get_rolled_stack_sizes

router = APIRouter()


class EnvironmentStepRequestBody(BaseModel):
    env_id: int
    action: int
    action_how_much: float


@router.post("/environment/{env_id}/step",
             response_model=EnvironmentState,
             operation_id="step_environment")
async def step_environment(body: EnvironmentStepRequestBody, request: Request):
    n_players = request.app.backend.active_ens[body.env_id].env.N_SEATS
    if body.action == -1:  # query ai model
        action = (0, -1)
    else:
        action = (body.action, body.action_how_much)
    # observation is always relative to
    obs, a, done, info = request.app.backend.active_ens[body.env_id].step(action)

    # offset relative to hero offset
    button_index = request.app.backend.metadata[body.env_id]['button_index']
    p_acts_next = request.app.backend.active_ens[body.env_id].env.current_player.seat_id
    offset = (p_acts_next + button_index) % n_players

    # if action was fold, but player could have checked, the environment internally changes the action
    # if that happens, we must overwrite last action accordingly
    action = request.app.backend.active_ens[body.env_id].env.last_action  # [what, how_much, who]
    action = action[0], action[1]  # drop who
    print(f'a = {a}')
    print(f'done = {done}')
    print(f'info = {info}')
    obs_dict = request.app.backend.active_ens[body.env_id].obs_idx_dict
    obs_keys = [k for k in obs_dict.keys()]

    table_info = get_table_info(obs_keys, obs, offset)
    idx_end_table = obs_keys.index('side_pot_5')

    board_cards = get_board_cards(idx_board_start=obs_keys.index('0th_board_card_rank_0'),
                                  idx_board_end=obs_keys.index('0th_player_card_0_rank_0'),
                                  obs=obs)
    # todo debug players using scratch
    player_info = get_player_stats(obs_keys, obs, start_idx=idx_end_table + 1, offset=offset, n_players=n_players)
    stack_sizes_rolled = get_rolled_stack_sizes(request, body, n_players, button_index)

    payouts_rolled = {}
    for k, v in info['payouts'].items():
        pid = (button_index + k) % n_players
        payouts_rolled[pid] = v


    # players_with_chips_left = [p if not p.is_all_in]
    result = {'env_id': body.env_id,
              'n_players': n_players,
              'stack_sizes': stack_sizes_rolled,
              'last_action': LastAction(**{'action_what': action[0], 'action_how_much': action[1]}),
              'table': table_info,
              'players': player_info,
              'board': board_cards,
              'button_index': offset,
              'done': done,
              # todo this jumps from 3 to 1 instead of going from 3 to 4
              'p_acts_next': (p_acts_next + button_index) % n_players,
              'info': Info(**{'continue_round': info['continue_round'],
                              'draw_next_stage': info['draw_next_stage'],
                              'rundown': info['rundown'],
                              'deal_next_hand': info['deal_next_hand'],
                              'payouts': payouts_rolled})
              }
    return EnvironmentState(**dict(result))
