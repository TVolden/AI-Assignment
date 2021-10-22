from typing import Sequence
from chess import *
from node import Node

PIECE_ORDER = [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING] # The order of the move pickers representations

def serialize(board: Board) -> Sequence[Node]:
    """
    Creates a board representation in 6 layers, one for each piece.
    White pieces are represented as 1 while black pieces are represented as -1
    The order of the layers are determined by the PIECE_ORDER
    """
    output = []
    for piece in PIECE_ORDER:
        w = board.pieces(piece, WHITE)
        b = board.pieces(piece, BLACK)
        output.extend([Node(1) if s in w else Node(-1) if s in b else Node(0) for s in reversed(range(64))])
    return output