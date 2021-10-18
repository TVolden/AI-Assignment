from __future__ import annotations
from typing import Sequence
import chess

class TreeNode:
    def __init__(self, state: chess.Board, action: chess.Move = None, parent: TreeNode = None):
        self.state:chess.Board = state
        self.actions:Sequence[chess.Move] = [m for m in state.legal_moves]
        self.action:chess.Move = action
        self.parent:TreeNode = parent
        self.result = 0
        self.explored = 0
        self.childNodes = dict[chess.Move, TreeNode]()

    def update_result(self, q: float):
        self.result += q
        #self.explored += 1
    
    def append_exploration(self):
        v = self
        while True:
            v.explored += 1
            if v.parent is None:
                break
            v = v.parent

    def explore(self) -> TreeNode:
        action = None
        for move in self.actions:
            if move not in self.childNodes.keys():
                action = move
                break
        board = chess.Board(self.state.fen())
        board.push(action)
        self.childNodes[action] = TreeNode(board, action, self)
        return self.childNodes[action]

    def get_average(self) -> float:
        return self.result / self.explored

    def get_explores(self) -> int:
        return self.explored
    
    def get_unexplored(self) -> int:
        return len(self.actions) - len(self.childNodes)

    def fully_explored(self) -> bool:
        return len(self.actions) == len(self.childNodes)

    def get_actions(self) -> Sequence[chess.Move]:
        return self.childNodes.keys()

    def get_child(self, action:chess.Move) -> TreeNode:
        return self.childNodes[action]

    def count_children(self) -> int:
        return len(self.childNodes)

    def get_state(self) -> chess.Board:
        return self.state
