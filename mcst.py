# Created by Thomas Volden
import math
import re
import time
from typing import Sequence
import chess
import random

from tree import TreeNode

class Policy:
    def pick_move(self, legal_moves: Sequence[chess.Move]) -> chess.Move:
        pass

class RandomPolicy(Policy):
    def pick_move(self, legal_moves: Sequence[chess.Move]) -> chess.Move:
        return random.choice(legal_moves)

class MonteCarloSearchTree:
    def __init__(self, exploration_weight=1, time_budget=1, policy: Policy = RandomPolicy()) -> None:
        self.policy = policy
        self.c = exploration_weight
        self.time_budget = time_budget

    def _simulate(self, node: TreeNode, player: chess.COLORS) -> float:
        "Returns the reward for a random simulation (to completion) of the board state"
        board = node.get_state()
        # Run until we hit a terminal state
        while not board.is_game_over():
            board.push(self.policy.pick_move([m for m in board.legal_moves]))
        # Return 0.5 if it's a draw, otherwise return 1 if current player won or 0 if other player won.
        return 0.5 if board.is_stalemate() else 1 if board.turn != player else 0

    def _expand(self, node: TreeNode):
        return node.explore()

    def _select(self, node: TreeNode):
        "find an unexplored descendent of the board state"
        v = node
        while not v.state.is_game_over():
            if not v.fully_explored():
                return self._expand(v)
            else:
                v = self._uct_select(v) # Best child
        return v

    def _backpropagate(self, node:TreeNode, reward):
        "Send the reward back up the ancestors of the leaf"
        v = node
        while True:
            v.update_result(reward)
            if v.parent is None:
                break
            v = v.parent
    
    def _uct_select(self, parent: TreeNode):
        "Select a child of state, balancing exploration & exploitation"
        
        best_result = float('-inf')
        best_child = None

        # Go through children to find the best option
        for action in parent.get_actions():
            child = parent.get_child(action)
            
            # Calculate a score
            # 1. Take the average from previous simulations (win:1, draw:0.5, loss=0)
            # 2. Add the upper confidence bound (√2ln(n)/nj)
            result = child.get_average() + self.c * math.sqrt((2*math.log(parent.get_explores()))/child.get_explores())
            
            # If this child score is the best current score
            if result > best_result:
                best_result = result
                best_child = child

        # Return the child with the best score
        return best_child

    def choose(self, board: chess.Board) -> chess.Move:
        "Choose a move in the game and execute it"
        v0 = v = TreeNode(board)
        start_time = time.time()
        while time.time() - start_time < self.time_budget:
            v = self._select(v0)
            value = self._simulate(v, board.turn)
            self._backpropagate(v, value)

        return self._uct_select(v0).action