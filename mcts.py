# Created by Thomas Volden
import math
import re
import time
from typing import Hashable, Sequence, Tuple
import chess
import random
from multiprocessing import Pool, cpu_count
import os

from tree import TreeNode

class Policy:
    def pick_move(self, board: chess.Board) -> chess.Move:
        pass

class RandomPolicy(Policy):
    def pick_move(self, board: chess.Board) -> chess.Move:
        return random.choice([m for m in board.legal_moves])

class DecisiveMovePolicy(Policy):
    def pick_move(self, board: chess.Board) -> chess.Move:
        moves = []
        for move in board.legal_moves:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return move
            board.pop()
            moves.append(move)

        return random.choice(moves)

class Proc:
    def __init__(self) -> None:
        self.node = None
        self.active = False
    
    def assign(self, node:TreeNode):
        self.active = True
        self.node = node

    def done(self):
        self.active = False

class MonteCarloTreeSearch:
    def __init__(self, exploration_weight=1, time_budget=1, policy: Policy = RandomPolicy(), parallel:bool = False) -> None:
        self.policy = policy
        self.c = exploration_weight
        self.time_budget = time_budget
        self.player = None
        self.parallel = parallel

    def _playouts(self, node:TreeNode, n_playouts:int = 64) -> float:
        #start_time = time.time()
        with Pool() as pool:
            scores = pool.map(self._simulate, [chess.Board(node.state.fen()) for _ in range(n_playouts)])
        #print("Simulation time %s sec" % ((time.time() - start_time)))
        return sum(scores)/len(scores)

    def _simulate(self, board: chess.Board) -> float:
        "Returns the reward for a random simulation (to completion) of the board state"
        # Run until we hit a terminal state
        while not board.is_game_over():
            board.push(self.policy.pick_move(board))
        # Return 0.5 if it's a draw, otherwise return 1 if current player won or 0 if other player won.
        #print("%s, " % (board.fullmove_number), end='')
        return 0.5 if board.is_stalemate() else 1 if board.outcome().winner == self.player else 0

    def _expand(self, node: TreeNode):
        return node.explore()

    def _select(self, node: TreeNode):
        "find an unexplored descendent of the board state"
        v = node
        while not v.state.is_game_over():
            if not v.fully_explored():
                output = self._expand(v)
                output.append_exploration() # Do it up front to affect the selection process early on
                return output
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
            result = child.get_average() + self.c * math.sqrt((2*math.log(parent.get_explores()+1))/(child.get_explores()+1))
            
            # If this child score is the best current score
            if result > best_result:
                best_result = result
                best_child = child

        # Return the child with the best score
        return best_child

    def _playout(self, node:TreeNode) -> Tuple[TreeNode, float]:
        return node, self._simulate(chess.Board(node.state.fen()))

    def _serial(self, v0):
        start_time = time.time()
        while time.time() - start_time < self.time_budget:
            v = self._select(v0)
            value = self._simulate(chess.Board(v.state.fen()))
            self._backpropagate(v, value)

    def _serial_parallel_playouts(self, v0):
        v = self._select(v0)
        value = self._playouts(v, os.cpu_count())
        self._backpropagate(v, value)

    def _proc_done(self, result):
        self._backpropagate(self.procs[result[0]].node, result[1])
        self.procs[result[0]].done()
    
    def _proc_sim(self, i:int, board:chess.Board) -> Tuple[int, float]:
        return i, self._simulate(chess.Board(board.fen()))

    def _parallel(self, v0):
        start_time = time.time()
        self.procs = [Proc() for _ in range(os.cpu_count())]
        with Pool() as pool:
            while time.time() - start_time < self.time_budget:
                for i, proc in enumerate(self.procs):
                    if not proc.active:
                        proc.assign(self._select(v0))
                        pool.apply_async(self._proc_sim, args=(i, proc.node.state), callback=self._proc_done)
                time.sleep(0.01)
            pool.close()
            pool.join()

    def choose(self, board: chess.Board) -> chess.Move:
        "Choose a move in the game and execute it"
        v0 = TreeNode(board)
        self.player = board.turn
        if self.parallel:
            self._parallel(v0)
        else:
            self._serial(v0)
        print("Explorations: %s" % (v0.get_explores()))
        return self._uct_select(v0).action