# Created by Thomas Volden
from typing import Sequence
import chess
import numpy as np
import signal

from mcst import MonteCarloSearchTree

def is_running(b:chess.Board) -> bool:
    return not (b.is_stalemate() or b.is_insufficient_material() or b.is_checkmate())

def human_input(b: chess.Board) -> chess.Move:
    move = None
    while move is None:
        san = input("Write move (SAN): ")
        try:
            move = b.parse_san(san)
        except:
            print("Illegal move, try again.")
            move = None
    return move

board = chess.Board()

def print_moves():
    print()
    print("---- Game moves ----")
    print(chess.Board().variation_san(board.move_stack)) 

mcst = MonteCarloSearchTree(time_budget=3)
players = (human_input, mcst.choose)
#players = (mcst.choose, mcst.choose)
turn = 1

def handler(signum, frame):
    print_moves()
    exit(1)

signal.signal(signal.SIGINT, handler)

while is_running(board):
    turn = 1 - turn
    print("---- Player %s's turn ----" %(turn+1))
    move = players[turn](board)
    print("Player %s moved: %s" %(turn+1, board.san(move)))
    board.push(move)
    print(board)
if board.is_stalemate():
    print("It's a draw!")
else:
    print("Player %s wins!" %(turn+1))

print_moves()