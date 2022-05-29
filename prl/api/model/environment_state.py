from typing import Optional, Dict, List

from pydantic import BaseModel


class Card(BaseModel):
    name: str
    suit: int
    rank: int
    index: int  # 0<=4 for board position


class Board(BaseModel):
    b0: Card
    b1: Card
    b2: Card
    b3: Card
    b4: Card


class Table(BaseModel):
    """"
    ante:   0.0
    small_blind:   0.0032992411870509386
    big_blind:   0.006598482374101877
    min_raise:   0.013196964748203754
    pot_amt:   0.0
    total_to_call:   0.006598482374101877
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
    """
    ante: float
    small_blind: float
    big_blind: float
    min_raise: float
    pot_amt: float
    total_to_call: float
    round_preflop: float
    round_flop: float
    round_turn: float
    round_river: float
    side_pot_0: float
    side_pot_1: float
    side_pot_2: float
    side_pot_3: float
    side_pot_4: float
    side_pot_5: float


class PlayerInfo(BaseModel):
    """"
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
    """
    pid: int
    stack_p: float
    curr_bet_p: float
    has_folded_this_episode_p: bool
    is_allin_p: bool
    side_pot_rank_p_is_0: int
    side_pot_rank_p_is_1: int
    side_pot_rank_p_is_2: int
    side_pot_rank_p_is_3: int
    side_pot_rank_p_is_4: int
    side_pot_rank_p_is_5: int
    c0: Card
    c1: Card


class LastAction(BaseModel):
    action_what: int
    action_how_much: float


class Info(BaseModel):
    continue_round: bool
    draw_next_stage: bool
    rundown: bool
    deal_next_hand: bool
    payouts: Optional[Dict[int, float]]


class Players(BaseModel):
    p0: PlayerInfo
    p1: PlayerInfo
    p2: PlayerInfo
    p3: PlayerInfo
    p4: PlayerInfo
    p5: PlayerInfo


class EnvironmentState(BaseModel):
    # meta
    env_id: int
    n_players: int
    stack_sizes: Dict
    # game
    table: Table
    players: Players
    board: Board
    button_index: int
    # utils
    last_action: Optional[LastAction]
    p_acts_next: int
    done: bool
    info: Info
