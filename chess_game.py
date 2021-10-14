# Created by Thomas Volden
from typing import Sequence
import chess

from mcst import MonteCarloSearchTree

def is_running(b:chess.Board) -> bool:
    return not (b.is_stalemate() or b.is_insufficient_material() or b.is_checkmate())

def human_input(b: chess.Board) -> chess.Move:
    uci = input("Write move (UCI): ")
    while chess.Move.from_uci(uci) not in b.legal_moves:
        print("Illegal move, try again")
        uci = input("Write move (UCI): ")
    return chess.Move.from_uci(uci)

board = chess.Board()
mcst = MonteCarloSearchTree(time_budget=3)
players = (human_input, mcst.choose)
turn = 1
while is_running(board):
    turn = 1 - turn
    print("---- Player %s's turn ----" %(turn+1))
    move = players[turn](board)
    board.push(move)
    print("Player %s moved: %s" %(turn+1, move))
    print(board)
if board.is_stalemate():
    print("It's a draw!")
else:
    print("Player %s wins!" %(turn+1))