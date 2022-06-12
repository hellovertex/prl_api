from random import randint

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel
from starlette.requests import Request

from prl.api.model.environment_state import EnvironmentState, LastAction, Info
from .utils import get_table_info, get_board_cards, get_player_stats, get_stacks

router = APIRouter()


class EnvironmentStepRequestBody(BaseModel):
    env_id: int
    action: int
    action_how_much: float


def get_action(request, body):
    if body.action == -1:  # query ai model, random action for now
        # todo query baseline TAG agent
        what = randint(0, 2)
        raise_amount = -1
        if what == 2:
            raise_amount = max(max([p.current_bet for p in request.app.backend.active_ens[body.env_id].env.seats]), 100)
        action = (what, raise_amount)
    else:
        action = (body.action, body.action_how_much)
    return action


@router.post("/environment/{env_id}/step",
             response_model=EnvironmentState,
             operation_id="step_environment")
async def step_environment(body: EnvironmentStepRequestBody, request: Request):
    env_id = body.env_id
    n_players = request.app.backend.active_ens[env_id].env.N_SEATS
    action = get_action(request, body)

    obs, a, done, info = request.app.backend.active_ens[env_id].step(action)
    # if action was fold, but player could have checked, the environment internally changes the action
    # if that happens, we must overwrite last action accordingly
    mapped_indices = request.app.backend.metadata[env_id]['mapped_indices']
    action = request.app.backend.active_ens[env_id].env.last_action  # [what, how_much, who]
    action = action[0], action[1], mapped_indices[action[2]]
    print(f'Stepping environment with action = {action}')

    pid_next_to_act_backend = request.app.backend.active_ens[env_id].env.current_player.seat_id
    offset_current_player_to_hero = pid_next_to_act_backend

    obs_dict = request.app.backend.active_ens[env_id].obs_idx_dict
    obs_keys = [k for k in obs_dict.keys()]
    normalization = request.app.backend.active_ens[env_id].normalization
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
    stack_sizes_rolled = get_stacks(player_info)
    payouts_rolled = {}
    print('info[payouts] = ', info['payouts'])
    for k, v in info['payouts'].items():
        pid = mapped_indices[int(k)]
        payouts_rolled[pid] = v

    # when done, the observation sets the stacks to 0
    if done:
        stack_sizes_rolled = request.app.backend.metadata[body.env_id]['last_stack_sizes']
        print('STACK_SIZS BEFORE APPLYING PAYOUTS:', stack_sizes_rolled)
        # for seat_id, (seat_pid, stack) in enumerate(stack_sizes_rolled.items()):
        #     if seat_id in payouts_rolled:
        #         stack_sizes_rolled[seat_pid] += payouts_rolled[seat_id]
        #     # manually subtract last action from players stack_size, environment does not do it
        #     if seat_id == action[2] and (action[0] != 0):
        #         stack_sizes_rolled[seat_pid] -= action[1]
        for i, player in enumerate(request.app.backend.active_ens[env_id].env.seats):
            stack_sizes_rolled[f'p{mapped_indices[i]}'] = player.stack
    request.app.backend.metadata[body.env_id]['last_stack_sizes'] = stack_sizes_rolled
    is_game_over = len(np.where(np.array(list(stack_sizes_rolled.values())) != 0)[0]) < 2
    print('done = ', done)
    print('RETURNING WITH STACK_SIZS:', stack_sizes_rolled)
    print('PASYOUS = ', payouts_rolled)
    # players_with_chips_left = [p if not p.is_all_in]
    result = {'env_id': body.env_id,
              'n_players': n_players,
              'stack_sizes': stack_sizes_rolled,
              'last_action': LastAction(**{'action_what': action[0], 'action_how_much': action[1]}),
              'table': table_info,
              'players': player_info,
              'board': board_cards,
              'button_index': request.app.backend.metadata[env_id]['button_index'],
              'sb': request.app.backend.metadata[env_id]['sb'],
              'bb': request.app.backend.metadata[env_id]['bb'],
              'done': done,
              'game_over': is_game_over,  # less than two players remaining
              'p_acts_next': mapped_indices[pid_next_to_act_backend],
              'info': Info(**{'continue_round': info['continue_round'],
                              'draw_next_stage': info['draw_next_stage'],
                              'rundown': info['rundown'],
                              'deal_next_hand': info['deal_next_hand'],
                              'payouts': payouts_rolled})
              }
    return EnvironmentState(**dict(result))
