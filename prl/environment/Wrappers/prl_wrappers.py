import enum
from typing import List, Optional

import numpy as np
from gym import spaces
from collections import defaultdict, deque
from prl.environment.steinberger.PokerRL.game.Poker import Poker

FOLD = 0
CHECK_CALL = 1
RAISE = 2


class Vectorizer:
    """ Abstract Vectorizer Interface. All vectorizers should be derived from this base class
    and implement the method "vectorize"."""

    def vectorize(self, obs, *args, **kwargs):
        """todo"""
        raise NotImplementedError


class AgentObservationType(enum.IntEnum):
    CARD_KNOWLEDGE = 1  # default where agent only sees his own cards and the board
    SEER = 2  # agent sees all player cards


class CanonicalVectorizer(Vectorizer):
    """Docstring"""

    # todo write vectorizer, such that it can be associated with exactly one env class
    def __init__(self,
                 num_players,
                 obs_idx_dict,
                 btn_pos,  # self.env.BTN_POS, always equal to 0
                 max_players=6,
                 n_ranks=13,
                 n_suits=4,
                 n_board_cards=5,
                 n_hand_cards=2,
                 use_zero_padding=True,
                 mode=AgentObservationType.CARD_KNOWLEDGE):
        # --- Utils --- #
        # todo [optional] switch to use num_players instead of max_players
        self._agent_observation_type = mode
        self._use_zero_padding = use_zero_padding
        self._next_player_who_gets_observation = None
        self.num_players = num_players
        self.obs_idx_dict = obs_idx_dict
        self._max_players = max_players
        self.n_ranks = n_ranks
        self.n_suits = n_suits
        self._n_board_cards = n_board_cards
        self._n_hand_cards = n_hand_cards
        self._n_stages = len(['preflop', 'flop', 'turn', 'river'])
        card = List[int]  # 2 cards
        hand = List[card]
        self._player_hands: Optional[List[hand]] = None
        self._action_history = None
        # btn_idx is equal to current player offset, since button is at index 0 inside environment
        # but we encode observation such that player is at index 0
        self._btn_idx = btn_pos
        max_actions_per_stage_per_player = 2
        max_actions = max_actions_per_stage_per_player * self._n_stages * self._max_players
        self._bits_per_action = len(['fold', 'check/call', 'bet/raise']) \
                                + len(['last_action_how_much'])
        self._bits_stats_per_player_original = len(['stack', 'curr_bet', 'has_folded', 'is_all_in']) \
                                               + len(['side_pot_rank_p0_is_']) * self.num_players
        self._bits_stats_per_player = len(['stack', 'curr_bet', 'has_folded', 'is_all_in']) \
                                      + len(['side_pot_rank_p0_is_']) * self._max_players
        self._bits_per_card = n_ranks + n_suits  # 13 ranks + 4 suits
        # --- Observation Bits --- #
        self._bits_table = len(['ante',
                                'small_blind',
                                'big_blind',
                                'min_raise',
                                'pot_amt',
                                'total_to_call'])
        self._bits_next_player = self._max_players
        self._bits_stage = self._n_stages
        self._bits_side_pots = self._max_players
        self._bits_player_stats = self._bits_stats_per_player * self._max_players
        self._bits_player_stats_original = self._bits_stats_per_player_original * self.num_players
        self._bits_board = self._bits_per_card * n_board_cards  # 3 cards flop, 1 card turn, 1 card river
        self._bits_player_hands = self._max_players * n_hand_cards * self._bits_per_card
        self._bits_action_history_one_player = max_actions_per_stage_per_player * self._n_stages * self._bits_per_action
        self._bits_action_history = self._bits_action_history_one_player * self._max_players

        # --- Offsets --- #
        self._start_table = 0
        self._start_next_player = self._bits_table
        self._start_stage = self._start_next_player + self._bits_next_player
        self._start_side_pots = self._start_stage + self._bits_stage
        self._start_player_stats = self._start_side_pots + self._bits_side_pots
        self._start_board = self._start_player_stats + self._bits_player_stats
        self._start_player_hands = self._start_board + self._bits_board
        self._start_action_history = self._start_player_hands + self._bits_player_hands
        self.offset = None

        # --- Number of features --- #
        self._obs_len = self._bits_table \
                        + self._bits_next_player \
                        + self._bits_stage \
                        + self._bits_side_pots \
                        + self._bits_player_stats \
                        + self._bits_board \
                        + self._bits_player_hands \
                        + self._bits_action_history
        self._obs = np.zeros(self._obs_len)

    @property
    def agent_observation_mode(self):
        return self._agent_observation_type

    @agent_observation_mode.setter
    def agent_observation_mode(self, mode: AgentObservationType):
        self._agent_observation_type = mode

    def vectorized_observation_shape(self):
        return self._obs.shape

    def encode_table(self, obs):
        """Example:
                    ante:   0.0
             small_blind:   0.05
               big_blind:   0.1
               min_raise:   0.2
                 pot_amt:   0.0
           total_to_call:   0.1
        """

        self.offset = 0 + self._bits_table
        assert self.offset == self._start_next_player
        # copy unchanged
        self._obs[0:self.offset] = obs[0:self.offset]

    def encode_next_player(self, obs):
        """Example:
            p0_acts_next:   1.0
            p1_acts_next:   0.0
            p2_acts_next:   0.0
          Since this is relative to observer, p0_acts_next will always be 1.
          We could remove this but keep it for consistency with the Steinberger Env.
        """
        self.offset += self._bits_next_player
        assert self.offset == self._start_stage
        # original obs indices
        start_orig = self.obs_idx_dict['p0_acts_next']
        end_orig = start_orig + self.num_players
        # extract from original observation
        bits = obs[start_orig:end_orig]
        bits = np.roll(bits, -self._next_player_who_gets_observation)
        # zero padding
        bits = np.pad(bits, (0, self._max_players - self.num_players), 'constant')
        # copy from original observation with zero padding
        self._obs[self._start_next_player:self.offset] = bits

    def encode_stage(self, obs):
        """Example:
           round_preflop:   1.0
              round_flop:   0.0
              round_turn:   0.0
             round_river:   0.0
        """
        self.offset += self._bits_stage
        assert self.offset == self._start_side_pots
        # original obs indices
        start_orig = self.obs_idx_dict['round_preflop']
        end_orig = start_orig + self._n_stages
        # extract from original observation
        bits = obs[start_orig:end_orig]
        # zero padding is not necessary
        # copy from original observation without zero padding
        self._obs[self._start_stage:self.offset] = bits

    def encode_side_pots(self, obs):
        """Example:
            side_pot_0:   0.0
            side_pot_1:   0.0
            side_pot_2:   0.0
        """
        self.offset += self._bits_side_pots
        assert self.offset == self._start_player_stats
        # original obs indices
        try:
            # if side pots are available in original observation
            start_orig = self.obs_idx_dict['side_pot_0']
            end_orig = start_orig + self.num_players
            bits = obs[start_orig:end_orig]
        except Exception as e:
            bits = np.zeros(self.num_players)

        # move self to index 0
        bits = np.roll(bits, -self._next_player_who_gets_observation)

        # zero padding
        bits = np.pad(bits, (0, self._max_players - self.num_players), 'constant')

        # copy from original observation with zero padding
        self._obs[self._start_side_pots:self.offset] = bits

    def encode_player_stats(self, obs):
        """Example:
        stack_p0:   0.9
                 curr_bet_p0:   0.1
    has_folded_this_episode_p0:   0.0
                 is_allin_p0:   0.0
       side_pot_rank_p0_is_0:   0.0
       side_pot_rank_p0_is_...:   0.0
       side_pot_rank_p0_is_n:   0.0

                    stack_p1:   0.95
                 curr_bet_p1:   0.05
    has_folded_this_episode_p1:   0.0
                 is_allin_p1:   0.0
       side_pot_rank_p1_is_0:   0.0
       side_pot_rank_p1_is_...:   0.0
       side_pot_rank_p1_is_n:   0.0
       """
        self.offset += self._bits_player_stats
        assert self.offset == self._start_board
        # original obs indices
        start_orig = self.obs_idx_dict['stack_p0']
        end_orig = start_orig + self._bits_player_stats_original
        # extract from original observation
        bits = np.array(obs[start_orig:end_orig])
        # zero padding in between players for additional side pot info
        bits_per_player = np.split(bits, self.num_players)
        bits_to_pad_in_between = np.zeros(self._max_players - self.num_players)
        padded_in_between = np.array([np.append(s, bits_to_pad_in_between) for s in bits_per_player])
        # roll
        roll_by = -self._next_player_who_gets_observation
        padded_in_between = np.roll(padded_in_between, roll_by, axis=0).flatten()
        # zero padding until the end
        padded_in_between = np.pad(padded_in_between, (0, self._bits_player_stats - len(padded_in_between)), 'constant')

        # copy from original observation with zero padding
        self._obs[self._start_player_stats:self.offset] = padded_in_between

    def encode_board(self, obs):
        """Example:
        0th_board_card_rank_0:   0.0
       0th_board_card_rank_1:   0.0
       0th_board_card_rank_2:   0.0
       0th_board_card_rank_3:   0.0
       0th_board_card_rank_4:   0.0
       0th_board_card_rank_5:   0.0
       0th_board_card_rank_6:   0.0
       0th_board_card_rank_7:   0.0
       0th_board_card_rank_8:   0.0
       0th_board_card_rank_9:   0.0
      0th_board_card_rank_10:   0.0
      0th_board_card_rank_11:   0.0
      0th_board_card_rank_12:   0.0
       0th_board_card_suit_0:   0.0
       0th_board_card_suit_1:   0.0
       0th_board_card_suit_2:   0.0
       0th_board_card_suit_3:   0.0
       1th_board_card_rank_0:   0.0
       1th_board_card_rank_1:   0.0
       1th_board_card_rank_2:   0.0
       1th_board_card_rank_3:   0.0
       1th_board_card_rank_4:   0.0
       1th_board_card_rank_5:   0.0
       1th_board_card_rank_6:   0.0
       1th_board_card_rank_7:   0.0
       1th_board_card_rank_8:   0.0
       1th_board_card_rank_9:   0.0
      1th_board_card_rank_10:   0.0
      1th_board_card_rank_11:   0.0
      1th_board_card_rank_12:   0.0
       1th_board_card_suit_0:   0.0
       1th_board_card_suit_1:   0.0
       1th_board_card_suit_2:   0.0
       1th_board_card_suit_3:   0.0
       2th_board_card_rank_0:   0.0
       2th_board_card_rank_1:   0.0
       2th_board_card_rank_2:   0.0
       2th_board_card_rank_3:   0.0
       2th_board_card_rank_4:   0.0
       2th_board_card_rank_5:   0.0
       2th_board_card_rank_6:   0.0
       2th_board_card_rank_7:   0.0
       2th_board_card_rank_8:   0.0
       2th_board_card_rank_9:   0.0
      2th_board_card_rank_10:   0.0
      2th_board_card_rank_11:   0.0
      2th_board_card_rank_12:   0.0
       2th_board_card_suit_0:   0.0
       2th_board_card_suit_1:   0.0
       2th_board_card_suit_2:   0.0
       2th_board_card_suit_3:   0.0
       3th_board_card_rank_0:   0.0
       3th_board_card_rank_1:   0.0
       3th_board_card_rank_2:   0.0
       3th_board_card_rank_3:   0.0
       3th_board_card_rank_4:   0.0
       3th_board_card_rank_5:   0.0
       3th_board_card_rank_6:   0.0
       3th_board_card_rank_7:   0.0
       3th_board_card_rank_8:   0.0
       3th_board_card_rank_9:   0.0
      3th_board_card_rank_10:   0.0
      3th_board_card_rank_11:   0.0
      3th_board_card_rank_12:   0.0
       3th_board_card_suit_0:   0.0
       3th_board_card_suit_1:   0.0
       3th_board_card_suit_2:   0.0
       3th_board_card_suit_3:   0.0
       4th_board_card_rank_0:   0.0
       4th_board_card_rank_1:   0.0
       4th_board_card_rank_2:   0.0
       4th_board_card_rank_3:   0.0
       4th_board_card_rank_4:   0.0
       4th_board_card_rank_5:   0.0
       4th_board_card_rank_6:   0.0
       4th_board_card_rank_7:   0.0
       4th_board_card_rank_8:   0.0
       4th_board_card_rank_9:   0.0
      4th_board_card_rank_10:   0.0
      4th_board_card_rank_11:   0.0
      4th_board_card_rank_12:   0.0
       4th_board_card_suit_0:   0.0
       4th_board_card_suit_1:   0.0
       4th_board_card_suit_2:   0.0
       4th_board_card_suit_3:   0.0
       """
        self.offset += self._bits_board
        assert self.offset == self._start_player_hands
        # original obs indices
        start_orig = self.obs_idx_dict['0th_board_card_rank_0']
        end_orig = start_orig + self._bits_board
        # extract from original observation
        bits = obs[start_orig:end_orig]
        # zero padding is not necessary
        # copy from original observation without zero padding
        self._obs[self._start_board:self.offset] = bits

    def encode_player_hands(self, obs):
        """Example:"""
        self.offset += self._bits_player_hands
        assert self.offset == self._start_action_history
        # move own cards to index 0
        roll_by = -self._next_player_who_gets_observation
        rolled_cards = np.roll(self._player_hands, roll_by, axis=0).reshape(-1, self._n_hand_cards)
        # rolled_cards = [[ 5  3], [ 5  0], [12  0], [ 9  1], [ -127  -127], [ -127  -127]]
        # replace NAN with 0
        rolled_cards[np.where(rolled_cards == Poker.CARD_NOT_DEALT_TOKEN_1D)] = 0
        if not self._agent_observation_type == AgentObservationType.SEER:
            # ignore all other players cards -> the agent should not see these
            rolled_cards = rolled_cards[:2]
        # rolled_cards = [[ 5  3], [ 5  0], [12  0], [ 9  1], [ 0  0], [ 0  0]]

        # initialize hand_bits to 0
        card_bits = self.n_ranks + self.n_suits
        # hand_bits = [0] * self._n_hand_cards * self.num_players * card_bits
        hand_bits = [0] * self._bits_player_hands
        # overwrite one_hot card_bits
        print(f'NUM_PLAYERS INSIDE VEC = {self.num_players}')
        for n_card, card in enumerate(rolled_cards):
            offset = card_bits * n_card
            # set rank
            hand_bits[card[0] + offset] = 1
            # set suit
            hand_bits[card[1] + offset + self.n_ranks] = 1
        # zero padding
        # hand_bits = np.pad(hand_bits, (0,  - len(hand_bits)), 'constant')
        # hand_bits = np.resize(hand_bits, self._bits_player_hands)
        self._obs[self._start_player_hands:self.offset] = hand_bits

    @staticmethod
    def _vectorize_deque(deq, normalization):
        bits_per_action = 4  # ==len([IS_FOLD, IS_CHECK_CALL, IS_RAISE, ACTION_HOW_MUCH])
        bits = [0 for _ in range(deq.maxlen * bits_per_action)]
        for i, action in enumerate(deq):
            how_much = 0 if not action[0] == RAISE else action[1] / normalization
            idx_action = 1 + action[0]
            bits[i * bits_per_action] = how_much
            bits[i * bits_per_action + idx_action] = 1
        return bits

    def encode_action_history(self, normalization):
        """Example:"""
        self.offset += self._bits_action_history
        assert self.offset == self._obs_len
        d = self._action_history.deque
        _bits_action_history = 192
        pids = [i for i in range(self.num_players)]
        pids = np.roll(pids, - self._next_player_who_gets_observation)
        bits = []
        # iterate all players, get preflop actions
        # iterate all players, get flop actions
        # iterate all players, get turn actions
        # iterate all players, get river actions
        for stage in ['preflop', 'flop', 'turn', 'river']:
            for pid in pids:
                bits.append(self._vectorize_deque(d[pid][stage], normalization))
            # append 0-bits for missing players
            for _ in range(self._max_players - self.num_players):
                bits.append([0, 0, 0, 0] * self._action_history.buffer_actions_per_stage())

        # pad flattened numpy array with zeros
        bits = np.array([bit for byte in bits for bit in byte])
        # bits = np.pad(bits, (0, self._bits_action_history - len(bits)), 'constant')
        self._obs[self._start_action_history:self.offset] = bits

    def vectorize(self, obs, _next_player_who_gets_observation=None, action_history=None, player_hands=None,
                  normalization=None):
        # reset
        self._obs = np.zeros(self._obs_len)
        self.offset = None
        self._player_hands = player_hands
        self._action_history = action_history
        self._next_player_who_gets_observation = _next_player_who_gets_observation
        # encode
        self.encode_table(obs)
        self.encode_next_player(obs)
        self.encode_stage(obs)
        self.encode_side_pots(obs)
        self.encode_player_stats(obs)
        self.encode_board(obs)
        self.encode_player_hands(obs)
        self.encode_action_history(normalization)
        # append button relative to observer
        # [4,5,0,1,2,3] -> btn_idx = 2
        # [1,2,3,4,5,0] -> btn_idx = 5
        assert self.offset == self._obs_len
        # append offset to button at the end as a hotfix
        return np.concatenate([self._obs, [self._next_player_who_gets_observation]])


class Positions6Max(enum.IntEnum):
    """Positions as in the literature, for a table with at most 6 Players.
BTN for Button, SB for Small Blind, etc...
"""
    BTN = 0
    SB = 1
    BB = 2
    UTG = 3  # UnderTheGun
    MP = 4  # Middle Position
    CO = 5  # CutOff


class Wrapper:

    def __init__(self, env):
        """
        Args:
            env:   The environment instance to be wrapped
        """
        self.env = env

    def reset(self, config):
        """Reset the environment with a new config.
        Signals environment handlers to reset and restart the environment using
        a config dict.
        Args:
          config: dict, specifying the parameters of the environment to be
            generated. May contain state_dict to generate a deterministic environment.
        Returns:
          observation: A dict containing the full observation state.
        """
        raise NotImplementedError("Not implemented in Abstract Base class")

    def step(self, action):
        """Take one step in the game.
        Args:
          action: object, mapping to an action taken by an agent.
        Returns:
          observation: object, Containing full observation state.
          reward: float, Reward obtained from taking the action.
          done: bool, Whether the game is done.
          info: dict, Optional debugging information.
        Raises:
          AssertionError: When an illegal action is provided.
        """
        raise NotImplementedError("Not implemented in Abstract Base class")


class WrapperPokerRL(Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self._player_hands = []

    def reset(self, config=None):
        """
        Resets the state of the game to the standard beginning of the episode. If specified in the args passed,
        stack size randomization is applied in the new episode. If deck_state_dict is not None, the cards
        and associated random variables are synchronized FROM the given environment, so that when .step() is called on
        each of them, they produce the same result.

        Args:
            config["deck_state_dict"]:      Optional.
                                            If an instance of a PokerEnv subclass is passed, the deck, holecards, and
                                            board in this instance will be synchronized from the handed env cls.
        """
        # assert config.get('deck_state_dict')
        self._before_reset(config)
        deck_state_dict = None
        if config is not None:
            deck_state_dict = config['deck_state_dict']
        env_obs, rew_for_all_players, done, info = self.env.reset(deck_state_dict=deck_state_dict)
        if not self._player_hands:
            for i in range(self.env.N_SEATS):
                self._player_hands.append(self.env.get_hole_cards_of_player(i))
        self._after_reset()

        return self._return_obs(env_obs=env_obs, rew_for_all_players=rew_for_all_players, done=done, info=info)

    def step(self, action):
        """
        Steps the environment from an action of the natural action representation to the environment.

        Returns:
            obs, reward, done, info
        """

        # callbacks in derived class
        self._before_step(action)

        # step environment
        env_obs, rew_for_all_players, done, info = self.env.step(action)

        self._after_step(action)
        # call get_current_obs of derived class
        return self._return_obs(env_obs=env_obs, rew_for_all_players=rew_for_all_players, done=done, info=info)

    def step_from_processed_tuple(self, action):
        """
        Steps the environment from a tuple (action, num_chips,).

        Returns:
            obs, reward, done, info
        """
        return self.step(action)

    def step_raise_pot_frac(self, pot_frac):
        """
        Steps the environment from a fractional pot raise instead of an action as usually specified.

        Returns:
            obs, reward, done, info
        """
        processed_action = (2, self.env.get_fraction_of_pot_raise(
            fraction=pot_frac, player_that_bets=self.env.current_player))

        return self.step(processed_action)

    def _return_obs(self, rew_for_all_players, done, info, env_obs=None):
        return self.get_current_obs(env_obs=env_obs), rew_for_all_players, done, info

    # _______________________________ Override to augment observation ________________________________

    def _before_step(self, action):
        raise NotImplementedError

    def _before_reset(self, config):
        raise NotImplementedError

    def _after_step(self, action):
        raise NotImplementedError

    def _after_reset(self):
        raise NotImplementedError

    def get_current_obs(self, env_obs):
        raise NotImplementedError

    # Can add additional callbacks here if necessary...


class ActionSpace(enum.IntEnum):
    """Under Construction"""
    FOLD = 0
    CHECK = 1
    CALL = 2
    RAISE_MIN_OR_3BB = 3
    RAISE_HALF_POT = 4
    RAISE_POT = 5
    ALL_IN = 6
    SMALL_BLIND = 7
    BIG_BLIND = 8


class ActionHistory:
    # noinspection PyTypeChecker
    def __init__(self, max_players, max_actions_per_player_per_stage):
        self._max_players = max_players
        self._max_actions_per_player_per_stage = max_actions_per_player_per_stage
        self.deque = {}

        for pos in range(self._max_players):
            # create default dictionary for current player for each stage
            # default dictionary stores only the last two actions per stage per player
            self.deque[Positions6Max(pos).value] = defaultdict(
                lambda: deque(maxlen=max_actions_per_player_per_stage),
                keys=['preflop', 'flop', 'turn', 'river'])

    def buffer_actions_per_stage(self):
        return self._max_actions_per_player_per_stage

    def __str__(self):
        representation = ""
        for player_seat_id in range(self._max_players):
            representation += f'--- Player {player_seat_id}:---\n' + \
                              f'Actions preflop: {self.deque[player_seat_id]["preflop"]}\n' + \
                              f'Actions flop: {self.deque[player_seat_id]["flop"]}\n' + \
                              f'Actions turn: {self.deque[player_seat_id]["turn"]}\n' + \
                              f'Actions river: {self.deque[player_seat_id]["river"]}\n'
        return representation

    def __repr__(self):
        return self.__str__()


class ActionHistoryWrapper(WrapperPokerRL):
    """Intermediate wrapper that pushes back each action into a history buffer,
    before passing it to the AugmentObservationWrapper"""

    def __init__(self, env):
        """
        Args:
            env (PokerEnv subclass instance):   The environment instance to be wrapped
        """
        super().__init__(env=env)
        self._player_hands = []
        self._rounds = ['preflop', 'flop', 'turn', 'river']
        self._actions_per_stage = ActionHistory(max_players=6, max_actions_per_player_per_stage=2)

        self._next_player_who_gets_observation = None
        # experimental
        self._actions_per_stage_discretized = ActionHistory(max_players=6, max_actions_per_player_per_stage=2)

    # _______________________________ Overridden ________________________________
    def _before_step(self, action):
        """
        """
        # store action in history buffer
        self._pushback_action(action,
                              player_who_acted=self.env.current_player.seat_id,
                              in_which_stage=self.env.current_round)

    def _after_step(self, action):
        """Called before observation is computed by vectorizer"""
        self._next_player_who_gets_observation = self.env.current_player.seat_id

    def _before_reset(self, config=None):
        """Called before observation is computed by vectorizer"""
        if config is not None and 'deck_state_dict' in config:
            if 'hand' in config['deck_state_dict']:
                # key 'hand' is set, when text files are parsed to vectorized observations
                self._player_hands = config['deck_state_dict']['hand']

    def _after_reset(self):
        self._next_player_who_gets_observation = self.env.current_player.seat_id

    # _______________________________ Action History ________________________________

    def discretize(self, action_formatted):
        if action_formatted[0] == 2:  # action is raise
            pot_size = self.env.get_all_winnable_money()
            raise_amt = action_formatted[1]
            if raise_amt < pot_size / 2:
                return ActionSpace.RAISE_MIN_OR_3BB
            elif raise_amt < pot_size:
                return ActionSpace.RAISE_HALF_POT
            elif raise_amt < 2 * pot_size:
                return ActionSpace.RAISE_POT
            else:
                return ActionSpace.ALL_IN
        else:  # action is fold or check/call
            action_discretized = action_formatted[0]
        return action_discretized

    def _pushback_action(self, action_formatted, player_who_acted, in_which_stage):
        # part of observation
        self._actions_per_stage.deque[player_who_acted][
            self._rounds[in_which_stage]].append(action_formatted)

        action_discretized = self.discretize(action_formatted)
        # for the neural network labels
        self._actions_per_stage_discretized.deque[player_who_acted][
            self._rounds[in_which_stage]].append(action_discretized)

    # _______________________________ Override to Augment observation ________________________________
    def get_current_obs(self, env_obs):
        """Implement this to encode Action History into observation"""
        raise NotImplementedError


# noinspection DuplicatedCode
class AugmentObservationWrapper(ActionHistoryWrapper):
    """Runs our custom vectorizer after computing the observation from the steinberger env"""

    def __init__(self, env):
        super().__init__(env=env)
        # todo: (?) check how obs is normalized to avoid small floats
        self.normalization = float(
            sum([s.starting_stack_this_episode for s in self.env.seats])
        ) / self.env.N_SEATS
        self.num_players = env.N_SEATS
        self.max_players = 6
        self.num_board_cards = 5
        self.observation_space, self.obs_idx_dict, self.obs_parts_idxs_dict = self._construct_obs_space()
        self._vectorizer = CanonicalVectorizer(num_players=self.num_players,
                                               obs_idx_dict=self.env.obs_idx_dict,
                                               # btn pos used to return obs relative to self
                                               btn_pos=self.env.BTN_POS)

    def overwrite_args(self, args, agent_observation_mode=None, n_players=None):
        if n_players:
            self.num_players = n_players
        self.env.set_args(args)
        if not agent_observation_mode:
            agent_observation_mode = self.env.agent_observation_mode()
        self.normalization = float(
            sum([s.starting_stack_this_episode for s in self.env.seats])
        ) / self.env.N_SEATS
        self.num_players = self.env.N_SEATS
        self.observation_space, self.obs_idx_dict, self.obs_parts_idxs_dict = self._construct_obs_space()
        self._vectorizer = CanonicalVectorizer(num_players=self.num_players,
                                               obs_idx_dict=self.env.obs_idx_dict,
                                               # btn pos used to return obs relative to self
                                               btn_pos=self.env.BTN_POS)
        self._vectorizer.agent_observation_mode = agent_observation_mode

    def agent_observation_mode(self):
        return self._vectorizer.agent_observation_mode

    def set_agent_observation_mode(self, mode: AgentObservationType):
        self._vectorizer.agent_observation_mode = mode

    def get_current_obs(self, env_obs):
        """
        Args:
            env_obs: the observation returned by the base PokerEnv.
            The len(env_obs) is a function of the number of players.
        """
        obs = self._vectorizer.vectorize(env_obs, self._next_player_who_gets_observation,
                                         action_history=self._actions_per_stage,
                                         player_hands=self._player_hands, normalization=self.normalization)
        # self.print_augmented_obs(obs)
        return obs

    def get_legal_actions(self):
        return self.env.get_legal_actions()

    def get_info(self):
        return self.env.get_info()

    def _construct_obs_space(self):
        """
        The maximum all chip-values can reach is n_seats, because we normalize by dividing by the average starting stack
        """
        obs_idx_dict = {}
        obs_parts_idxs_dict = {
            "board": [],
            "players": [[] for _ in range(self.max_players)],
            "table_state": [],
            "player_cards": [[] for _ in range(self.max_players)],
            "action_history": [[] for _ in range(len(self._rounds))]
        }
        next_idx = [0]  # list is a mutatable object. int not.

        def get_discrete(size, name, _curr_idx):
            obs_idx_dict[name] = _curr_idx[0]
            _curr_idx[0] += 1
            return spaces.Discrete(size)

        def get_new_box(name, _curr_idx, high, low=0):
            obs_idx_dict[name] = _curr_idx[0]
            _curr_idx[0] += 1
            return spaces.Box(low=low, high=high, shape=(1,), dtype=np.float32)

        # __________________________  Public Information About Game State  _________________________
        _k = next_idx[0]
        _table_space = [  # (blinds are in obs to give the agent a perspective on starting stack after normalization
            get_new_box("ante", next_idx, self.max_players),  # .................................... self.ANTE
            get_new_box("small_blind", next_idx, self.max_players),  # ............................. self.SMALL_BLIND
            get_new_box("big_blind", next_idx, self.max_players),  # ............................... self.BIG_BLIND
            get_new_box("min_raise", next_idx, self.max_players),  # ............................... min total raise
            get_new_box("pot_amt", next_idx, self.max_players),  # ................................. main_pot amount
            get_new_box("total_to_call", next_idx, self.max_players),  # ........................... total_to_call
            # get_new_box("last_action_how_much", next_idx, self.max_players),  # .................... self.last_action[1]
        ]
        # for i in range(3):  # .................................................................. self.last_action[0]
        #   _table_space.append(get_discrete(1, "last_action_what_" + str(i), next_idx))
        #
        # for i in range(self.max_players):  # ....................................................... self.last_action[2]
        #   _table_space.append(get_discrete(1, "last_action_who_" + str(i), next_idx))

        for i in range(self.max_players):  # ....................................................... curr_player.seat_id
            _table_space.append(get_discrete(1, "p" + str(i) + "_acts_next", next_idx))

        for i in range(len(self._rounds)):  # ...................................... round onehot
            _table_space.append(get_discrete(1, "round_" + Poker.INT2STRING_ROUND[i], next_idx)),

        for i in range(self.max_players):  # ....................................................... side pots
            _table_space.append(get_new_box("side_pot_" + str(i), next_idx, 1))

        # add to parts_dict for possible slicing for agents.
        obs_parts_idxs_dict["table_state"] += list(range(_k, next_idx[0]))

        # __________________________  Public Information About Each Player  ________________________
        _player_space = []
        for i in range(self.max_players):
            _k = next_idx[0]
            _player_space += [
                get_new_box("stack_p" + str(i), next_idx, self.max_players),  # ..................... stack
                get_new_box("curr_bet_p" + str(i), next_idx, self.max_players),  # .................. current_bet
                get_discrete(1, "has_folded_this_episode_p" + str(i), next_idx),  # ............. folded_this_epis
                get_discrete(1, "is_allin_p" + str(i), next_idx),  # ............................ is_allin
            ]
            for j in range(self.max_players):
                _player_space.append(
                    get_discrete(1, "side_pot_rank_p" + str(i) + "_is_" + str(j), next_idx))  # . side_pot_rank

            # add to parts_dict for possible slicing for agents
            obs_parts_idxs_dict["players"][i] += list(range(_k, next_idx[0]))

        # _______________________________  Public cards (i.e. board)  ______________________________
        _board_space = []
        _k = next_idx[0]
        for i in range(self.num_board_cards):

            x = []
            for j in range(self.env.N_RANKS):  # .................................................... rank
                x.append(get_discrete(1, str(i) + "th_board_card_rank_" + str(j), next_idx))

            for j in range(self.env.N_SUITS):  # .................................................... suit
                x.append(get_discrete(1, str(i) + "th_board_card_suit_" + str(j), next_idx))

            _board_space += x

        # add to parts_dict for possible slicing for agents
        obs_parts_idxs_dict["board"] += list(range(_k, next_idx[0]))

        # _______________________________  Private Cards (i.e. players hands)  ______________________________
        # add to parts_dict for possible slicing for agents
        _handcards_space = []

        for i in range(self.max_players):
            _k = next_idx[0]
            for k in range(self.env.N_HOLE_CARDS):
                x = []
                for j in range(self.env.N_RANKS):  # .................................................... rank
                    x.append(get_discrete(1, str(i) + f"th_player_card_{k}_rank_" + str(j), next_idx))

                for j in range(self.env.N_SUITS):  # .................................................... suit
                    x.append(get_discrete(1, str(i) + f"th_player_card_{k}_suit_" + str(j), next_idx))

                _handcards_space += x

            obs_parts_idxs_dict["player_cards"][i] += list(range(_k, next_idx[0]))
        # _______________________________  Action History (max 2 /stage/player)  ______________________________
        # add to parts_dict for possible slicing for agents
        # preflop_player_0_action_0_how_much
        # preflop_player_0_action_0_what_0
        # preflop_player_0_action_0_what_1
        # preflop_player_0_action_0_what_2
        # preflop_player_0_action_1_how_much
        # preflop_player_0_action_1_what_0
        # preflop_player_0_action_1_what_1
        # preflop_player_0_action_1_what_2
        _action_history_space = []

        for i in range(len(self._rounds)):
            _k = next_idx[0]
            for j in range(self.max_players):
                for a in [0, 1]:
                    _action_history_space.append(
                        get_new_box(f"{self._rounds[i]}_player_{j}_action_{a}_how_much", next_idx, self.max_players))
                    for k in range(3):
                        _action_history_space.append(
                            get_discrete(1, f"{self._rounds[i]}_player_{j}_action_{a}_what_{k}", next_idx)
                        )
            obs_parts_idxs_dict["action_history"][i] += list(range(_k, next_idx[0]))
        # preflop, flop, turn, river : [action0, action1], []

        # __________________________  Return Complete _Observation Space  __________________________
        # Tuple (lots of spaces.Discrete and spaces.Box)
        _observation_space = spaces.Tuple(_table_space + _player_space + _board_space)
        _observation_space.shape = [len(_observation_space.spaces)]

        return _observation_space, obs_idx_dict, obs_parts_idxs_dict

    def print_augmented_obs(self, obs):
        """Can be used for debugging. Printed Observation example:
        ______________________________________ Printing _Observation _________________________________________
                              ante:   0.0
                       small_blind:   0.0032992411870509386
                         big_blind:   0.006598482374101877
                         min_raise:   0.013196964748203754
                           pot_amt:   0.0
                     total_to_call:   0.006598482374101877
                      p0_acts_next:   1.0
                      p1_acts_next:   0.0
                      p2_acts_next:   0.0
                      p3_acts_next:   1.0
                      p4_acts_next:   0.0
                      p5_acts_next:   0.0
                     round_preflop:   1.0
                        round_flop:   0.0
                        round_turn:   0.0
                       round_river:   0.0
                        side_pot_0:   0.0
                        side_pot_1:   0.0
                        side_pot_2:   0.0
                        side_pot_3:   0.0
                        side_pot_4:   0.0
                        side_pot_5:   0.0
                          stack_p0:   1.106103539466858
                       curr_bet_p0:   0.0
        has_folded_this_episode_p0:   0.0
                       is_allin_p0:   0.0
             side_pot_rank_p0_is_0:   0.0
             side_pot_rank_p0_is_1:   0.0
             side_pot_rank_p0_is_2:   0.0
             side_pot_rank_p0_is_3:   0.0
             side_pot_rank_p0_is_4:   0.0
             side_pot_rank_p0_is_5:   0.0
                          stack_p1:   0.5709006786346436
                       curr_bet_p1:   0.0032992411870509386
        has_folded_this_episode_p1:   0.0
                       is_allin_p1:   0.0
             side_pot_rank_p1_is_0:   0.0
             side_pot_rank_p1_is_1:   0.0
             side_pot_rank_p1_is_2:   0.0
             side_pot_rank_p1_is_3:   0.0
             side_pot_rank_p1_is_4:   0.0
             side_pot_rank_p1_is_5:   0.0
                          stack_p2:   1.3130979537963867
                       curr_bet_p2:   0.006598482374101877
        has_folded_this_episode_p2:   0.0
                       is_allin_p2:   0.0
             side_pot_rank_p2_is_0:   0.0
             side_pot_rank_p2_is_1:   0.0
             side_pot_rank_p2_is_2:   0.0
             side_pot_rank_p2_is_3:   0.0
             side_pot_rank_p2_is_4:   0.0
             side_pot_rank_p2_is_5:   0.0
                          stack_p3:   1.106103539466858
                       curr_bet_p3:   0.0
        has_folded_this_episode_p3:   0.0
                       is_allin_p3:   0.0
             side_pot_rank_p3_is_0:   0.0
             side_pot_rank_p3_is_1:   0.0
             side_pot_rank_p3_is_2:   0.0
             side_pot_rank_p3_is_3:   0.0
             side_pot_rank_p3_is_4:   0.0
             side_pot_rank_p3_is_5:   0.0
                          stack_p4:   0.5709006786346436
                       curr_bet_p4:   0.0032992411870509386
        has_folded_this_episode_p4:   0.0
                       is_allin_p4:   0.0
             side_pot_rank_p4_is_0:   0.0
             side_pot_rank_p4_is_1:   0.0
             side_pot_rank_p4_is_2:   0.0
             side_pot_rank_p4_is_3:   0.0
             side_pot_rank_p4_is_4:   0.0
             side_pot_rank_p4_is_5:   0.0
                          stack_p5:   1.3130979537963867
                       curr_bet_p5:   0.006598482374101877
        has_folded_this_episode_p5:   0.0
                       is_allin_p5:   0.0
             side_pot_rank_p5_is_0:   0.0
             side_pot_rank_p5_is_1:   0.0
             side_pot_rank_p5_is_2:   0.0
             side_pot_rank_p5_is_3:   0.0
             side_pot_rank_p5_is_4:   0.0
             side_pot_rank_p5_is_5:   0.0
             0th_board_card_rank_0:   0.0
             0th_board_card_rank_1:   0.0
             0th_board_card_rank_2:   0.0
             0th_board_card_rank_3:   0.0
             0th_board_card_rank_4:   0.0
             0th_board_card_rank_5:   0.0
             0th_board_card_rank_6:   0.0
             0th_board_card_rank_7:   0.0
             0th_board_card_rank_8:   0.0
             0th_board_card_rank_9:   0.0
            0th_board_card_rank_10:   0.0
            0th_board_card_rank_11:   0.0
            0th_board_card_rank_12:   0.0
             0th_board_card_suit_0:   0.0
             0th_board_card_suit_1:   0.0
             0th_board_card_suit_2:   0.0
             0th_board_card_suit_3:   0.0
             1th_board_card_rank_0:   0.0
             1th_board_card_rank_1:   0.0
             1th_board_card_rank_2:   0.0
             1th_board_card_rank_3:   0.0
             1th_board_card_rank_4:   0.0
             1th_board_card_rank_5:   0.0
             1th_board_card_rank_6:   0.0
             1th_board_card_rank_7:   0.0
             1th_board_card_rank_8:   0.0
             1th_board_card_rank_9:   0.0
            1th_board_card_rank_10:   0.0
            1th_board_card_rank_11:   0.0
            1th_board_card_rank_12:   0.0
             1th_board_card_suit_0:   0.0
             1th_board_card_suit_1:   0.0
             1th_board_card_suit_2:   0.0
             1th_board_card_suit_3:   0.0
             2th_board_card_rank_0:   0.0
             2th_board_card_rank_1:   0.0
             2th_board_card_rank_2:   0.0
             2th_board_card_rank_3:   0.0
             2th_board_card_rank_4:   0.0
             2th_board_card_rank_5:   0.0
             2th_board_card_rank_6:   0.0
             2th_board_card_rank_7:   0.0
             2th_board_card_rank_8:   0.0
             2th_board_card_rank_9:   0.0
            2th_board_card_rank_10:   0.0
            2th_board_card_rank_11:   0.0
            2th_board_card_rank_12:   0.0
             2th_board_card_suit_0:   0.0
             2th_board_card_suit_1:   0.0
             2th_board_card_suit_2:   0.0
             2th_board_card_suit_3:   0.0
             3th_board_card_rank_0:   0.0
             3th_board_card_rank_1:   0.0
             3th_board_card_rank_2:   0.0
             3th_board_card_rank_3:   0.0
             3th_board_card_rank_4:   0.0
             3th_board_card_rank_5:   0.0
             3th_board_card_rank_6:   0.0
             3th_board_card_rank_7:   0.0
             3th_board_card_rank_8:   0.0
             3th_board_card_rank_9:   0.0
            3th_board_card_rank_10:   0.0
            3th_board_card_rank_11:   0.0
            3th_board_card_rank_12:   0.0
             3th_board_card_suit_0:   0.0
             3th_board_card_suit_1:   0.0
             3th_board_card_suit_2:   0.0
             3th_board_card_suit_3:   0.0
             4th_board_card_rank_0:   0.0
             4th_board_card_rank_1:   0.0
             4th_board_card_rank_2:   0.0
             4th_board_card_rank_3:   0.0
             4th_board_card_rank_4:   0.0
             4th_board_card_rank_5:   0.0
             4th_board_card_rank_6:   0.0
             4th_board_card_rank_7:   0.0
             4th_board_card_rank_8:   0.0
             4th_board_card_rank_9:   0.0
            4th_board_card_rank_10:   0.0
            4th_board_card_rank_11:   0.0
            4th_board_card_rank_12:   0.0
             4th_board_card_suit_0:   0.0
             4th_board_card_suit_1:   0.0
             4th_board_card_suit_2:   0.0
             4th_board_card_suit_3:   0.0
          0th_player_card_0_rank_0:   0.0
          0th_player_card_0_rank_1:   0.0
          0th_player_card_0_rank_2:   0.0
          0th_player_card_0_rank_3:   0.0
          0th_player_card_0_rank_4:   0.0
          0th_player_card_0_rank_5:   1.0
          0th_player_card_0_rank_6:   0.0
          0th_player_card_0_rank_7:   0.0
          0th_player_card_0_rank_8:   0.0
          0th_player_card_0_rank_9:   0.0
         0th_player_card_0_rank_10:   0.0
         0th_player_card_0_rank_11:   0.0
         0th_player_card_0_rank_12:   0.0
           0th_board_card_0_suit_0:   0.0
           0th_board_card_0_suit_1:   0.0
           0th_board_card_0_suit_2:   0.0
           0th_board_card_0_suit_3:   1.0
          0th_player_card_1_rank_0:   0.0
          0th_player_card_1_rank_1:   0.0
          0th_player_card_1_rank_2:   0.0
          0th_player_card_1_rank_3:   0.0
          0th_player_card_1_rank_4:   0.0
          0th_player_card_1_rank_5:   1.0
          0th_player_card_1_rank_6:   0.0
          0th_player_card_1_rank_7:   0.0
          0th_player_card_1_rank_8:   0.0
          0th_player_card_1_rank_9:   0.0
         0th_player_card_1_rank_10:   0.0
         0th_player_card_1_rank_11:   0.0
         0th_player_card_1_rank_12:   0.0
           0th_board_card_1_suit_0:   1.0
           0th_board_card_1_suit_1:   0.0
           0th_board_card_1_suit_2:   0.0
           0th_board_card_1_suit_3:   0.0
          1th_player_card_0_rank_0:   0.0
          1th_player_card_0_rank_1:   0.0
          1th_player_card_0_rank_2:   0.0
          1th_player_card_0_rank_3:   0.0
          1th_player_card_0_rank_4:   0.0
          1th_player_card_0_rank_5:   0.0
          1th_player_card_0_rank_6:   0.0
          1th_player_card_0_rank_7:   0.0
          1th_player_card_0_rank_8:   0.0
          1th_player_card_0_rank_9:   0.0
         1th_player_card_0_rank_10:   0.0
         1th_player_card_0_rank_11:   0.0
         1th_player_card_0_rank_12:   1.0
           1th_board_card_0_suit_0:   1.0
           1th_board_card_0_suit_1:   0.0
           1th_board_card_0_suit_2:   0.0
           1th_board_card_0_suit_3:   0.0
          1th_player_card_1_rank_0:   0.0
          1th_player_card_1_rank_1:   0.0
          1th_player_card_1_rank_2:   0.0
          1th_player_card_1_rank_3:   0.0
          1th_player_card_1_rank_4:   0.0
          1th_player_card_1_rank_5:   0.0
          1th_player_card_1_rank_6:   0.0
          1th_player_card_1_rank_7:   0.0
          1th_player_card_1_rank_8:   0.0
          1th_player_card_1_rank_9:   1.0
         1th_player_card_1_rank_10:   0.0
         1th_player_card_1_rank_11:   0.0
         1th_player_card_1_rank_12:   0.0
           1th_board_card_1_suit_0:   0.0
           1th_board_card_1_suit_1:   1.0
           1th_board_card_1_suit_2:   0.0
           1th_board_card_1_suit_3:   0.0
          2th_player_card_0_rank_0:   1.0
          2th_player_card_0_rank_1:   0.0
          2th_player_card_0_rank_2:   0.0
          2th_player_card_0_rank_3:   0.0
          2th_player_card_0_rank_4:   0.0
          2th_player_card_0_rank_5:   0.0
          2th_player_card_0_rank_6:   0.0
          2th_player_card_0_rank_7:   0.0
          2th_player_card_0_rank_8:   0.0
          2th_player_card_0_rank_9:   0.0
         2th_player_card_0_rank_10:   0.0
         2th_player_card_0_rank_11:   0.0
         2th_player_card_0_rank_12:   0.0
           2th_board_card_0_suit_0:   1.0
           2th_board_card_0_suit_1:   0.0
           2th_board_card_0_suit_2:   0.0
           2th_board_card_0_suit_3:   0.0
          2th_player_card_1_rank_0:   1.0
          2th_player_card_1_rank_1:   0.0
          2th_player_card_1_rank_2:   0.0
          2th_player_card_1_rank_3:   0.0
          2th_player_card_1_rank_4:   0.0
          2th_player_card_1_rank_5:   0.0
          2th_player_card_1_rank_6:   0.0
          2th_player_card_1_rank_7:   0.0
          2th_player_card_1_rank_8:   0.0
          2th_player_card_1_rank_9:   0.0
         2th_player_card_1_rank_10:   0.0
         2th_player_card_1_rank_11:   0.0
         2th_player_card_1_rank_12:   0.0
           2th_board_card_1_suit_0:   1.0
           2th_board_card_1_suit_1:   0.0
           2th_board_card_1_suit_2:   0.0
           2th_board_card_1_suit_3:   0.0
          3th_player_card_0_rank_0:   0.0
          3th_player_card_0_rank_1:   0.0
          3th_player_card_0_rank_2:   0.0
          3th_player_card_0_rank_3:   0.0
          3th_player_card_0_rank_4:   0.0
          3th_player_card_0_rank_5:   1.0
          3th_player_card_0_rank_6:   0.0
          3th_player_card_0_rank_7:   0.0
          3th_player_card_0_rank_8:   0.0
          3th_player_card_0_rank_9:   0.0
         3th_player_card_0_rank_10:   0.0
         3th_player_card_0_rank_11:   0.0
         3th_player_card_0_rank_12:   0.0
           3th_board_card_0_suit_0:   0.0
           3th_board_card_0_suit_1:   0.0
           3th_board_card_0_suit_2:   0.0
           3th_board_card_0_suit_3:   1.0
          3th_player_card_1_rank_0:   0.0
          3th_player_card_1_rank_1:   0.0
          3th_player_card_1_rank_2:   0.0
          3th_player_card_1_rank_3:   0.0
          3th_player_card_1_rank_4:   0.0
          3th_player_card_1_rank_5:   1.0
          3th_player_card_1_rank_6:   0.0
          3th_player_card_1_rank_7:   0.0
          3th_player_card_1_rank_8:   0.0
          3th_player_card_1_rank_9:   0.0
         3th_player_card_1_rank_10:   0.0
         3th_player_card_1_rank_11:   0.0
         3th_player_card_1_rank_12:   0.0
           3th_board_card_1_suit_0:   1.0
           3th_board_card_1_suit_1:   0.0
           3th_board_card_1_suit_2:   0.0
           3th_board_card_1_suit_3:   0.0
          4th_player_card_0_rank_0:   0.0
          4th_player_card_0_rank_1:   0.0
          4th_player_card_0_rank_2:   0.0
          4th_player_card_0_rank_3:   0.0
          4th_player_card_0_rank_4:   0.0
          4th_player_card_0_rank_5:   0.0
          4th_player_card_0_rank_6:   0.0
          4th_player_card_0_rank_7:   0.0
          4th_player_card_0_rank_8:   0.0
          4th_player_card_0_rank_9:   0.0
         4th_player_card_0_rank_10:   0.0
         4th_player_card_0_rank_11:   0.0
         4th_player_card_0_rank_12:   1.0
           4th_board_card_0_suit_0:   1.0
           4th_board_card_0_suit_1:   0.0
           4th_board_card_0_suit_2:   0.0
           4th_board_card_0_suit_3:   0.0
          4th_player_card_1_rank_0:   0.0
          4th_player_card_1_rank_1:   0.0
          4th_player_card_1_rank_2:   0.0
          4th_player_card_1_rank_3:   0.0
          4th_player_card_1_rank_4:   0.0
          4th_player_card_1_rank_5:   0.0
          4th_player_card_1_rank_6:   0.0
          4th_player_card_1_rank_7:   0.0
          4th_player_card_1_rank_8:   0.0
          4th_player_card_1_rank_9:   1.0
         4th_player_card_1_rank_10:   0.0
         4th_player_card_1_rank_11:   0.0
         4th_player_card_1_rank_12:   0.0
           4th_board_card_1_suit_0:   0.0
           4th_board_card_1_suit_1:   1.0
           4th_board_card_1_suit_2:   0.0
           4th_board_card_1_suit_3:   0.0
          5th_player_card_0_rank_0:   1.0
          5th_player_card_0_rank_1:   0.0
          5th_player_card_0_rank_2:   0.0
          5th_player_card_0_rank_3:   0.0
          5th_player_card_0_rank_4:   0.0
          5th_player_card_0_rank_5:   0.0
          5th_player_card_0_rank_6:   0.0
          5th_player_card_0_rank_7:   0.0
          5th_player_card_0_rank_8:   0.0
          5th_player_card_0_rank_9:   0.0
         5th_player_card_0_rank_10:   0.0
         5th_player_card_0_rank_11:   0.0
         5th_player_card_0_rank_12:   0.0
           5th_board_card_0_suit_0:   1.0
           5th_board_card_0_suit_1:   0.0
           5th_board_card_0_suit_2:   0.0
           5th_board_card_0_suit_3:   0.0
          5th_player_card_1_rank_0:   1.0
          5th_player_card_1_rank_1:   0.0
          5th_player_card_1_rank_2:   0.0
          5th_player_card_1_rank_3:   0.0
          5th_player_card_1_rank_4:   0.0
          5th_player_card_1_rank_5:   0.0
          5th_player_card_1_rank_6:   0.0
          5th_player_card_1_rank_7:   0.0
          5th_player_card_1_rank_8:   0.0
          5th_player_card_1_rank_9:   0.0
         5th_player_card_1_rank_10:   0.0
         5th_player_card_1_rank_11:   0.0
         5th_player_card_1_rank_12:   0.0
           5th_board_card_1_suit_0:   1.0
           5th_board_card_1_suit_1:   0.0
           5th_board_card_1_suit_2:   0.0
           5th_board_card_1_suit_3:   0.0
preflop_player_0_action_0_how_much:   0.0
  preflop_player_0_action_0_what_0:   0.0
  preflop_player_0_action_0_what_1:   0.0
  preflop_player_0_action_0_what_2:   0.0
preflop_player_0_action_1_how_much:   0.0
  preflop_player_0_action_1_what_0:   0.0
  preflop_player_0_action_1_what_1:   0.0
  preflop_player_0_action_1_what_2:   0.0
preflop_player_1_action_0_how_much:   0.0
  preflop_player_1_action_0_what_0:   0.0
  preflop_player_1_action_0_what_1:   0.0
  preflop_player_1_action_0_what_2:   0.0
preflop_player_1_action_1_how_much:   0.0
  preflop_player_1_action_1_what_0:   0.0
  preflop_player_1_action_1_what_1:   0.0
  preflop_player_1_action_1_what_2:   0.0
preflop_player_2_action_0_how_much:   0.0
  preflop_player_2_action_0_what_0:   0.0
  preflop_player_2_action_0_what_1:   0.0
  preflop_player_2_action_0_what_2:   0.0
preflop_player_2_action_1_how_much:   0.0
  preflop_player_2_action_1_what_0:   0.0
  preflop_player_2_action_1_what_1:   0.0
  preflop_player_2_action_1_what_2:   0.0
preflop_player_3_action_0_how_much:   0.0
  preflop_player_3_action_0_what_0:   0.0
  preflop_player_3_action_0_what_1:   0.0
  preflop_player_3_action_0_what_2:   0.0
preflop_player_3_action_1_how_much:   0.0
  preflop_player_3_action_1_what_0:   0.0
  preflop_player_3_action_1_what_1:   0.0
  preflop_player_3_action_1_what_2:   0.0
preflop_player_4_action_0_how_much:   0.0
  preflop_player_4_action_0_what_0:   0.0
  preflop_player_4_action_0_what_1:   0.0
  preflop_player_4_action_0_what_2:   0.0
preflop_player_4_action_1_how_much:   0.0
  preflop_player_4_action_1_what_0:   0.0
  preflop_player_4_action_1_what_1:   0.0
  preflop_player_4_action_1_what_2:   0.0
preflop_player_5_action_0_how_much:   0.0
  preflop_player_5_action_0_what_0:   0.0
  preflop_player_5_action_0_what_1:   0.0
  preflop_player_5_action_0_what_2:   0.0
preflop_player_5_action_1_how_much:   0.0
  preflop_player_5_action_1_what_0:   0.0
  preflop_player_5_action_1_what_1:   0.0
  preflop_player_5_action_1_what_2:   0.0
   flop_player_0_action_0_how_much:   0.0
     flop_player_0_action_0_what_0:   0.0
     flop_player_0_action_0_what_1:   0.0
     flop_player_0_action_0_what_2:   0.0
   flop_player_0_action_1_how_much:   0.0
     flop_player_0_action_1_what_0:   0.0
     flop_player_0_action_1_what_1:   0.0
     flop_player_0_action_1_what_2:   0.0
   flop_player_1_action_0_how_much:   0.0
     flop_player_1_action_0_what_0:   0.0
     flop_player_1_action_0_what_1:   0.0
     flop_player_1_action_0_what_2:   0.0
   flop_player_1_action_1_how_much:   0.0
     flop_player_1_action_1_what_0:   0.0
     flop_player_1_action_1_what_1:   0.0
     flop_player_1_action_1_what_2:   0.0
   flop_player_2_action_0_how_much:   0.0
     flop_player_2_action_0_what_0:   0.0
     flop_player_2_action_0_what_1:   0.0
     flop_player_2_action_0_what_2:   0.0
   flop_player_2_action_1_how_much:   0.0
     flop_player_2_action_1_what_0:   0.0
     flop_player_2_action_1_what_1:   0.0
     flop_player_2_action_1_what_2:   0.0
   flop_player_3_action_0_how_much:   0.0
     flop_player_3_action_0_what_0:   0.0
     flop_player_3_action_0_what_1:   0.0
     flop_player_3_action_0_what_2:   0.0
   flop_player_3_action_1_how_much:   0.0
     flop_player_3_action_1_what_0:   0.0
     flop_player_3_action_1_what_1:   0.0
     flop_player_3_action_1_what_2:   0.0
   flop_player_4_action_0_how_much:   0.0
     flop_player_4_action_0_what_0:   0.0
     flop_player_4_action_0_what_1:   0.0
     flop_player_4_action_0_what_2:   0.0
   flop_player_4_action_1_how_much:   0.0
     flop_player_4_action_1_what_0:   0.0
     flop_player_4_action_1_what_1:   0.0
     flop_player_4_action_1_what_2:   0.0
   flop_player_5_action_0_how_much:   0.0
     flop_player_5_action_0_what_0:   0.0
     flop_player_5_action_0_what_1:   0.0
     flop_player_5_action_0_what_2:   0.0
   flop_player_5_action_1_how_much:   0.0
     flop_player_5_action_1_what_0:   0.0
     flop_player_5_action_1_what_1:   0.0
     flop_player_5_action_1_what_2:   0.0
   turn_player_0_action_0_how_much:   0.0
     turn_player_0_action_0_what_0:   0.0
     turn_player_0_action_0_what_1:   0.0
     turn_player_0_action_0_what_2:   0.0
   turn_player_0_action_1_how_much:   0.0
     turn_player_0_action_1_what_0:   0.0
     turn_player_0_action_1_what_1:   0.0
     turn_player_0_action_1_what_2:   0.0
   turn_player_1_action_0_how_much:   0.0
     turn_player_1_action_0_what_0:   0.0
     turn_player_1_action_0_what_1:   0.0
     turn_player_1_action_0_what_2:   0.0
   turn_player_1_action_1_how_much:   0.0
     turn_player_1_action_1_what_0:   0.0
     turn_player_1_action_1_what_1:   0.0
     turn_player_1_action_1_what_2:   0.0
   turn_player_2_action_0_how_much:   0.0
     turn_player_2_action_0_what_0:   0.0
     turn_player_2_action_0_what_1:   0.0
     turn_player_2_action_0_what_2:   0.0
   turn_player_2_action_1_how_much:   0.0
     turn_player_2_action_1_what_0:   0.0
     turn_player_2_action_1_what_1:   0.0
     turn_player_2_action_1_what_2:   0.0
   turn_player_3_action_0_how_much:   0.0
     turn_player_3_action_0_what_0:   0.0
     turn_player_3_action_0_what_1:   0.0
     turn_player_3_action_0_what_2:   0.0
   turn_player_3_action_1_how_much:   0.0
     turn_player_3_action_1_what_0:   0.0
     turn_player_3_action_1_what_1:   0.0
     turn_player_3_action_1_what_2:   0.0
   turn_player_4_action_0_how_much:   0.0
     turn_player_4_action_0_what_0:   0.0
     turn_player_4_action_0_what_1:   0.0
     turn_player_4_action_0_what_2:   0.0
   turn_player_4_action_1_how_much:   0.0
     turn_player_4_action_1_what_0:   0.0
     turn_player_4_action_1_what_1:   0.0
     turn_player_4_action_1_what_2:   0.0
   turn_player_5_action_0_how_much:   0.0
     turn_player_5_action_0_what_0:   0.0
     turn_player_5_action_0_what_1:   0.0
     turn_player_5_action_0_what_2:   0.0
   turn_player_5_action_1_how_much:   0.0
     turn_player_5_action_1_what_0:   0.0
     turn_player_5_action_1_what_1:   0.0
     turn_player_5_action_1_what_2:   0.0
  river_player_0_action_0_how_much:   0.0
    river_player_0_action_0_what_0:   0.0
    river_player_0_action_0_what_1:   0.0
    river_player_0_action_0_what_2:   0.0
  river_player_0_action_1_how_much:   0.0
    river_player_0_action_1_what_0:   0.0
    river_player_0_action_1_what_1:   0.0
    river_player_0_action_1_what_2:   0.0
  river_player_1_action_0_how_much:   0.0
    river_player_1_action_0_what_0:   0.0
    river_player_1_action_0_what_1:   0.0
    river_player_1_action_0_what_2:   0.0
  river_player_1_action_1_how_much:   0.0
    river_player_1_action_1_what_0:   0.0
    river_player_1_action_1_what_1:   0.0
    river_player_1_action_1_what_2:   0.0
  river_player_2_action_0_how_much:   0.0
    river_player_2_action_0_what_0:   0.0
    river_player_2_action_0_what_1:   0.0
    river_player_2_action_0_what_2:   0.0
  river_player_2_action_1_how_much:   0.0
    river_player_2_action_1_what_0:   0.0
    river_player_2_action_1_what_1:   0.0
    river_player_2_action_1_what_2:   0.0
  river_player_3_action_0_how_much:   0.0
    river_player_3_action_0_what_0:   0.0
    river_player_3_action_0_what_1:   0.0
    river_player_3_action_0_what_2:   0.0
  river_player_3_action_1_how_much:   0.0
    river_player_3_action_1_what_0:   0.0
    river_player_3_action_1_what_1:   0.0
    river_player_3_action_1_what_2:   0.0
  river_player_4_action_0_how_much:   0.0
    river_player_4_action_0_what_0:   0.0
    river_player_4_action_0_what_1:   0.0
    river_player_4_action_0_what_2:   0.0
  river_player_4_action_1_how_much:   0.0
    river_player_4_action_1_what_0:   0.0
    river_player_4_action_1_what_1:   0.0
    river_player_4_action_1_what_2:   0.0
  river_player_5_action_0_how_much:   0.0
    river_player_5_action_0_what_0:   0.0
    river_player_5_action_0_what_1:   0.0
    river_player_5_action_0_what_2:   0.0
  river_player_5_action_1_how_much:   0.0
    river_player_5_action_1_what_0:   0.0
    river_player_5_action_1_what_1:   0.0
    river_player_5_action_1_what_2:   0.0"""
        print("______________________________________ Printing _Observation _________________________________________")
        names = [e + ":  " for e in list(self.obs_idx_dict.keys())]
        str_len = max([len(e) for e in names])
        for name, key in zip(names, list(self.obs_idx_dict.keys())):
            name = name.rjust(str_len)
            print(name, obs[self.obs_idx_dict[key]])

    @property
    def current_player(self):
        return self.env.current_player
