import enum
import json

import numpy as np
import requests
from fastapi import APIRouter
from pydantic import BaseModel
from starlette.requests import Request

from model.environment_state import EnvironmentState, LastAction, Info
from .utils import get_table_info, get_board_cards, get_player_stats, get_stacks

router = APIRouter()
FOLD = 0
CHECK_CALL = 1
RAISE = 2


class EnvironmentStepRequestBody(BaseModel):
    env_id: int
    action: int
    action_how_much: float


class BetSizes(enum.IntEnum):
    # different from ActionSpace in prl_environment because
    # model was trained with 0: Fold, 1:CHECK_CALL 2: MIN_RAISE
    # instead of ActionSpace labels that are 0: FOLD 1: CHECK 2: CALL 3: MIN_RAISE,...
    MIN_RAISE = 2
    HALF_POT = 3
    POT = 4
    ALL_IN = 5


def parse_int_action(request, body, action):
    """Translate discretized integer action to steinberger-formatted (action_what, action_how_much) format."""

    action_what = action if action in [FOLD, CHECK_CALL] else RAISE  # 0 is fold, 1 is call, 2 is raise
    action_how_much = -1  # default bet size for non-bet/raise moves
    if action_what == RAISE:
        min_raise = request.app.backend.active_ens[body.env_id].env._get_current_total_min_raise()
        pot_size = request.app.backend.active_ens[body.env_id].env.get_all_winnable_money()
        all_in = max([player.stack for player in request.app.backend.active_ens[body.env_id].env.seats])
        # environment automatically adjusts bet size appropriately so it is safe to simply use
        # the largest stack for all-in raise amount as there is no over-raise
        # this way we do not need to determine whose players turn it is
        if action == BetSizes.MIN_RAISE:
            action_how_much = min_raise
        elif action == BetSizes.HALF_POT:
            action_how_much = max(min_raise, .5 * pot_size)
        elif action == BetSizes.POT:
            action_how_much = max(min_raise, pot_size)
        elif action == BetSizes.ALL_IN:
            action_how_much = max(min_raise, all_in)
    return action_what, action_how_much


def get_action(request, body):
    if body.action == -1:  # query pytorch model
        aws_lambda_torch_model_url = "https://p235bek4niablvfxiktuwuswni0qecqs.lambda-url.eu-central-1.on.aws/"
        query = list(request.app.backend.metadata[body.env_id]['last_obs'])
        model_output = requests.post(url=aws_lambda_torch_model_url,
                                     data=json.dumps({"query": query}),
                                     headers={'Content-Type': 'application/json'})
        int_action = eval(model_output.text)['action']
        return parse_int_action(request, body, int_action)
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
    request.app.backend.metadata[env_id]['last_obs'] = obs
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
    # todo: remove last_stack_sizes entirely and replace with stacks from seats
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
              'last_action': LastAction(**{'action_what': action[0],
                                           'action_how_much': action[1],
                                           'action_who': action[2]}),
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
