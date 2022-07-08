# 2. roll stack sizes
import numpy as np

from prl.environment.Wrappers.prl_wrappers import ActionHistory, AgentObservationType
from prl.environment.tests.utils import make_wrapped_env

DEFAULT_STARTING_STACK_SIZE = 20000


# def vectorize_deque(deq, normalization):
#     bits_per_action = 4
#     bits = [0 for _ in range(deq.maxlen * bits_per_action)]
#     for i, action in enumerate(deq):
#         bits[i * bits_per_action + action[0]] = 1
#         bits[i * bits_per_action + 3] = action[1] / normalization
#     return bits
#
#
# def vectorize_aoh_relative_to_observer(aoh, _next_player_who_gets_observation):
#     normalization = 1
#     d = aoh.deque
#     _max_players = 6
#     _bits_action_history = 192
#     pids = [i for i in range(_max_players)]
#     pids = np.roll(pids, - _next_player_who_gets_observation)
#     bits = []
#
#     # iterate all players, get preflop actions
#     # iterate all players, get flop actions
#     # iterate all players, get turn actions
#     # iterate all players, get river actions
#     for stage in ['preflop', 'flop', 'turn', 'river']:
#         for pid in pids:
#             bits.append(vectorize_deque(d[pid][stage], normalization))
#     return bits
#
#
# def test_vectorizer_action_history_vectorization():
#     aoh = ActionHistory(max_players=6, max_actions_per_player_per_stage=2)
#     seat = 0
#     stage = 'preflop'
#     aoh.deque[seat][stage].append((2, 100))
#     vec = vectorize_aoh_relative_to_observer(aoh, _next_player_who_gets_observation=1)
#     print(vec)
#     aoh.deque[seat + 1][stage].append((0, 100))
#     vec = vectorize_aoh_relative_to_observer(aoh, _next_player_who_gets_observation=0)
#     print(vec)
#     # print([v for vc in vec for v in vc])


# create env with 2,3,6 players and step until river and assert action history for all of them
def test_two_player_action_history_preflop():
    env = make_wrapped_env(n_players=2,
                           starting_stack_sizes=[DEFAULT_STARTING_STACK_SIZE for _ in range(2)])
    obs, _, _, _ = env.reset()
    # get relevant bits from vectorized information
    obs_keys = [k for k in env.obs_idx_dict.keys()]
    start = obs_keys.index('preflop_player_0_action_0_how_much')
    end = obs_keys.index('river_player_5_action_1_what_2') + 1
    # aoh_keys = obs_keys[obs_keys.index('preflop_player_0_action_0_how_much'):obs_keys.index(
    #     'river_player_5_action_0_what_2') + 1]
    # import pprint
    # pprint.pprint(dict(list(zip(aoh_keys, obs[start:end]))))
    # p0 raises
    obs, _, _, _ = env.step((2, 200))
    bits = obs[start:end]

    # Assert p0 action is rolled with observer offset (n_players-1)
    assert obs[obs_keys.index('preflop_player_1_action_0_what_2')] == 1
    assert sum([bit for bit in bits if bit == 1]) == 1

    # p1 calls
    obs, _, _, _ = env.step((1, -1))
    bits = obs[start:end]

    # Assert p1 action is rolled with observer offset (n_players-1)
    # p1 action must be at index 0 because he starts immediately after the flop
    assert obs[obs_keys.index('preflop_player_0_action_0_what_1')] == 1
    assert sum([bit for bit in bits if bit == 1]) == 2

    # Assert players postflop bits  are still zero
    bits_flop_and_later = obs[obs_keys.index('flop_player_0_action_0_how_much'):end]
    assert sum([bit for bit in bits_flop_and_later if bit == 1]) == 0

    # p1 raises first in flop after calling
    obs, _, _, _ = env.step((2, 400))
    assert obs[obs_keys.index('flop_player_1_action_0_what_2')] == 1


def test_two_player_action_history_flop():
    env = make_wrapped_env(n_players=2,
                           starting_stack_sizes=[DEFAULT_STARTING_STACK_SIZE for _ in range(2)])
    obs, _, _, _ = env.reset()
    # get relevant bits from vectorized information
    obs_keys = [k for k in env.obs_idx_dict.keys()]
    # p0 check/calls
    obs, _, _, _ = env.step((1, -1))
    # p1 check/calls
    obs, _, _, _ = env.step((1, -1))
    # p0 check/calls in FLOP
    obs, _, _, _ = env.step((1, -1))
    start = obs_keys.index('preflop_player_0_action_0_how_much')
    end = obs_keys.index('river_player_5_action_1_what_2') + 1
    bits = obs[start:end]

    assert sum([bit for bit in bits if bit == 1]) == 3
    assert obs[obs_keys.index('flop_player_1_action_0_what_1')] == 1


def test_player_cards(env):
    obs, _, _, _ = env.reset()
    obs_keys = [k for k in env.obs_idx_dict.keys()]
    # Assert one-hot encoded player hand
    start = obs_keys.index('0th_player_card_0_rank_0')
    end = obs_keys.index('1th_player_card_0_rank_0')
    bits = obs[start:end]
    assert sum([bit for bit in bits if bit == 1]) == 4  # (suit, rank) one-hot for two cards
    # Assert hidden other player cards
    start = obs_keys.index('1th_player_card_0_rank_0')
    end = obs_keys.index('5th_player_card_1_suit_3') + 1
    bits = obs[start:end]
    obs_mode = env.agent_observation_mode()
    vec = env._vectorizer
    assert sum([bit for bit in bits if bit == 1]) == 0  # other players cards are hidden


def test_player_cards_seer(env):
    env.set_agent_observation_mode(AgentObservationType.SEER)
    obs, _, _, _ = env.reset()
    obs_keys = [k for k in env.obs_idx_dict.keys()]
    start = obs_keys.index('0th_player_card_0_rank_0')
    end = obs_keys.index('5th_player_card_1_suit_3') + 1
    bits = obs[start:end]
    # Assert all players cards are one-hot encoded in SEER mode
    assert sum([bit for bit in bits if bit == 1]) == 4 * env.num_players


def test_three_player_action_history():
    env = make_wrapped_env(n_players=3,
                           starting_stack_sizes=[DEFAULT_STARTING_STACK_SIZE for _ in range(3)])
    obs, _, _, _ = env.reset()
    # get relevant bits from vectorized information
    obs_keys = [k for k in env.obs_idx_dict.keys()]

    # -- PREFLOP --
    # p0 raises 200
    obs, _, _, _ = env.step((2, 200))
    assert obs[obs_keys.index('preflop_player_2_action_0_what_2')] == 1  # relative to p1
    # p1 calls
    obs, _, _, _ = env.step((1, -1))
    assert obs[obs_keys.index('preflop_player_1_action_0_what_2')] == 1  # relative to p2
    assert obs[obs_keys.index('preflop_player_2_action_0_what_1')] == 1
    # p2 calls
    obs, _, _, _ = env.step((1, -1))
    assert obs[obs_keys.index('preflop_player_0_action_0_what_1')] == 1  # relative to p1
    assert obs[obs_keys.index('preflop_player_1_action_0_what_1')] == 1
    assert obs[obs_keys.index('preflop_player_2_action_0_what_2')] == 1

    # -- FLOP --
    # p1 raises 400
    obs, _, _, _ = env.step((2, 400))
    assert obs[obs_keys.index('preflop_player_0_action_0_what_1')] == 1  # relative to p2
    assert obs[obs_keys.index('preflop_player_1_action_0_what_2')] == 1
    assert obs[obs_keys.index('preflop_player_2_action_0_what_1')] == 1
    assert obs[obs_keys.index('flop_player_2_action_0_what_2')] == 1

    # p2 calls
    obs, _, _, _ = env.step((1, -1))
    assert obs[obs_keys.index('preflop_player_0_action_0_what_2')] == 1  # relative to p0
    assert obs[obs_keys.index('preflop_player_1_action_0_what_1')] == 1
    assert obs[obs_keys.index('preflop_player_2_action_0_what_1')] == 1
    assert obs[obs_keys.index('flop_player_1_action_0_what_2')] == 1
    assert obs[obs_keys.index('flop_player_2_action_0_what_1')] == 1

    # p0 calls
    obs, _, _, _ = env.step((1, -1))
    assert obs[obs_keys.index('preflop_player_0_action_0_what_1')] == 1  # relative to p1
    assert obs[obs_keys.index('preflop_player_1_action_0_what_1')] == 1
    assert obs[obs_keys.index('preflop_player_2_action_0_what_2')] == 1
    assert obs[obs_keys.index('flop_player_0_action_0_what_2')] == 1
    assert obs[obs_keys.index('flop_player_1_action_0_what_1')] == 1
    assert obs[obs_keys.index('flop_player_2_action_0_what_1')] == 1

    # -- TURN --
    # p1 checks
    obs, _, _, _ = env.step((1, -1))
    assert obs[obs_keys.index('preflop_player_0_action_0_what_1')] == 1  # relative to p2
    assert obs[obs_keys.index('preflop_player_1_action_0_what_2')] == 1
    assert obs[obs_keys.index('preflop_player_2_action_0_what_1')] == 1
    assert obs[obs_keys.index('flop_player_0_action_0_what_1')] == 1
    assert obs[obs_keys.index('flop_player_1_action_0_what_1')] == 1
    assert obs[obs_keys.index('flop_player_2_action_0_what_2')] == 1
    assert obs[obs_keys.index('turn_player_2_action_0_what_1')] == 1
    # p2 raises 800
    obs, _, _, _ = env.step((2, 800))
    assert obs[obs_keys.index('preflop_player_0_action_0_what_2')] == 1  # relative to p0
    assert obs[obs_keys.index('preflop_player_1_action_0_what_1')] == 1
    assert obs[obs_keys.index('preflop_player_2_action_0_what_1')] == 1
    assert obs[obs_keys.index('flop_player_0_action_0_what_1')] == 1
    assert obs[obs_keys.index('flop_player_1_action_0_what_2')] == 1
    assert obs[obs_keys.index('flop_player_2_action_0_what_1')] == 1
    assert obs[obs_keys.index('turn_player_1_action_0_what_1')] == 1
    assert obs[obs_keys.index('turn_player_2_action_0_what_2')] == 1
    # p0 folds
    obs, _, _, _ = env.step((0, -1))
    assert obs[obs_keys.index('preflop_player_0_action_0_what_1')] == 1  # relative to p1
    assert obs[obs_keys.index('preflop_player_1_action_0_what_1')] == 1
    assert obs[obs_keys.index('preflop_player_2_action_0_what_2')] == 1
    assert obs[obs_keys.index('flop_player_0_action_0_what_2')] == 1
    assert obs[obs_keys.index('flop_player_1_action_0_what_1')] == 1
    assert obs[obs_keys.index('flop_player_2_action_0_what_1')] == 1
    assert obs[obs_keys.index('turn_player_0_action_0_what_1')] == 1
    assert obs[obs_keys.index('turn_player_1_action_0_what_2')] == 1
    assert obs[obs_keys.index('turn_player_2_action_0_what_0')] == 1
    # p1 calls
    obs, _, _, _ = env.step((1, -1))
    assert obs[obs_keys.index('turn_player_0_action_0_what_1')] == 1  # relative to p1
    assert obs[obs_keys.index('turn_player_0_action_1_what_1')] == 1
    # -- RIVER --
    # p1 checks
    obs, _, _, _ = env.step((1, -1))
    assert obs[obs_keys.index('river_player_0_action_0_what_0')] == 0  # relative to p2
    assert obs[obs_keys.index('river_player_0_action_0_what_1')] == 0
    assert obs[obs_keys.index('river_player_0_action_0_what_2')] == 0
    assert obs[obs_keys.index('river_player_0_action_1_what_0')] == 0
    assert obs[obs_keys.index('river_player_0_action_1_what_1')] == 0
    assert obs[obs_keys.index('river_player_0_action_1_what_2')] == 0
    assert obs[obs_keys.index('river_player_2_action_0_what_0')] == 0
    assert obs[obs_keys.index('river_player_2_action_0_what_1')] == 1
    assert obs[obs_keys.index('river_player_2_action_0_what_2')] == 0
    assert obs[obs_keys.index('river_player_2_action_1_what_0')] == 0
    assert obs[obs_keys.index('river_player_2_action_1_what_1')] == 0
    assert obs[obs_keys.index('river_player_2_action_1_what_2')] == 0
    # p2 raises
    obs, _, _, _ = env.step((2, 2000))
    assert obs[obs_keys.index('river_player_0_action_0_what_0')] == 0  # relative to p1
    assert obs[obs_keys.index('river_player_0_action_0_what_1')] == 1
    assert obs[obs_keys.index('river_player_0_action_0_what_2')] == 0
    assert obs[obs_keys.index('river_player_0_action_1_what_0')] == 0
    assert obs[obs_keys.index('river_player_0_action_1_what_1')] == 0
    assert obs[obs_keys.index('river_player_0_action_1_what_2')] == 0
    assert obs[obs_keys.index('river_player_1_action_0_what_0')] == 0
    assert obs[obs_keys.index('river_player_1_action_0_what_1')] == 0
    assert obs[obs_keys.index('river_player_1_action_0_what_2')] == 1
    assert obs[obs_keys.index('river_player_1_action_1_what_0')] == 0
    assert obs[obs_keys.index('river_player_1_action_1_what_1')] == 0
    assert obs[obs_keys.index('river_player_1_action_1_what_2')] == 0
    # p1 re-raises
    obs, _, _, _ = env.step((2, 6000))
    assert obs[obs_keys.index('river_player_0_action_0_what_0')] == 0  # relative to p2
    assert obs[obs_keys.index('river_player_0_action_0_what_1')] == 0
    assert obs[obs_keys.index('river_player_0_action_0_what_2')] == 1
    assert obs[obs_keys.index('river_player_0_action_1_what_0')] == 0
    assert obs[obs_keys.index('river_player_0_action_1_what_1')] == 0
    assert obs[obs_keys.index('river_player_0_action_1_what_2')] == 0
    assert obs[obs_keys.index('river_player_2_action_0_what_0')] == 0
    assert obs[obs_keys.index('river_player_2_action_0_what_1')] == 1
    assert obs[obs_keys.index('river_player_2_action_0_what_2')] == 0
    assert obs[obs_keys.index('river_player_2_action_1_what_0')] == 0
    assert obs[obs_keys.index('river_player_2_action_1_what_1')] == 0
    assert obs[obs_keys.index('river_player_2_action_1_what_2')] == 1
    # p2 re-re-raises
    obs, _, _, _ = env.step((2, 12000))
    assert obs[obs_keys.index('river_player_0_action_0_what_0')] == 0  # relative to p1
    assert obs[obs_keys.index('river_player_0_action_0_what_1')] == 1
    assert obs[obs_keys.index('river_player_0_action_0_what_2')] == 0
    assert obs[obs_keys.index('river_player_0_action_1_what_0')] == 0
    assert obs[obs_keys.index('river_player_0_action_1_what_1')] == 0
    assert obs[obs_keys.index('river_player_0_action_1_what_2')] == 1
    assert obs[obs_keys.index('river_player_1_action_0_what_0')] == 0
    assert obs[obs_keys.index('river_player_1_action_0_what_1')] == 0
    assert obs[obs_keys.index('river_player_1_action_0_what_2')] == 1
    assert obs[obs_keys.index('river_player_1_action_1_what_0')] == 0
    assert obs[obs_keys.index('river_player_1_action_1_what_1')] == 0
    assert obs[obs_keys.index('river_player_1_action_1_what_2')] == 1
    # p1 all in
    obs, _, _, _ = env.step((2, 20000))
    assert obs[obs_keys.index('river_player_0_action_0_what_2')] == 1  # relative to p2
    assert obs[obs_keys.index('river_player_0_action_1_what_2')] == 1
    assert obs[obs_keys.index('river_player_2_action_0_what_2')] == 1
    assert obs[obs_keys.index('river_player_2_action_1_what_2')] == 1


def test_six_player_action_history():
    # todo:
    # scenario 2: call/check until river
    # scenario 3: raise call until river
    # scenario 4: pre flop all ins -> rundown till showdown
    pass

def test_payouts():
    env = make_wrapped_env(n_players=3,
                           starting_stack_sizes=[2000 for _ in range(3)])
    obs, _, _, _ = env.reset()
    env.step(2,200)
    env.step(2,2000)
    env.step(0, -1)

if __name__ == '__main__':
    for n_players in range(2, 7):
        starting_stack_sizes = [DEFAULT_STARTING_STACK_SIZE for _ in range(n_players)]
        wrapped_env = make_wrapped_env(n_players, starting_stack_sizes)
        # test_two_player_action_history()
        test_player_cards(wrapped_env)
        test_player_cards_seer(wrapped_env)

    test_two_player_action_history_preflop()
    test_two_player_action_history_flop()
    test_three_player_action_history()
    test_payouts()