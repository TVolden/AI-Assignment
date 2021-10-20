import random
from typing import Dict
import numpy as np
from chess import polyglot, Move, Board
from policy import Policy

class QTable(Policy):
    def __init__(self, learning_rate = 0.1, discount_rate = 1.0, ε = 0.3) -> None:
        self.table:Dict[str, Dict[str, float]] = {}
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.ε = ε
        self.hasher = polyglot.ZobristHasher(polyglot.POLYGLOT_RANDOM_ARRAY)
    
    def _hash(self, state: Board) -> int:
        return self.hasher(state)

    def pick_move(self, state: Board) -> Move:
        s = str(self._hash(state))
        if s in self.table and random.uniform(0,1) > self.ε:
            actions = self.table[s]
            return Move.from_uci(max(actions, key=actions.get))

        return random.choice([m for m in state.legal_moves])

    def expected_reward(self, state: Board, action: Move) -> float:
        s = str(self._hash(state))
        a = action.uci()
        return self.table[s][a] if s in self.table and a in self.table[s] else 0

    def _max(self, state: Board) -> float:
        s = str(self._hash(state))
        if s not in self.table:
            return 0
        return max(self.table[s].values())

    def update(self, state: Board, action: Move, reward: float, new_state: Board) -> None:
        s = str(self._hash(state))
        a = action.uci()
        if (s not in self.table):
            self.table[s] = { }
        if (a not in self.table[s]):
            self.table[s][a] = 0

        self.table[s][a] = self.table[s][a] + self.learning_rate * (reward + self.discount_rate * self._max(new_state) - self.table[s][a])
