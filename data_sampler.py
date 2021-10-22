from math import exp
from typing import Dict, Sequence, Tuple
from node import Node, square
import random
import chess
import chess_utils

class DataSampler:
    def __init__(self, data: Dict[str, Dict[str, float]]) -> None:
        """
        data: Q-Table sample data
        """
        self.data = data

    def SamplePiecePickerData(self) -> Tuple[Sequence[Node], Sequence[Node]]:
        """
        Sample random data from the Q-Table to be used as training data for a piece picker network
        """
        state = random.choice(list(self.data)) # Select a random state
        board = chess.Board(state)# Restore the state from the FEN notation
        input = chess_utils.serialize(board) # Turn the state into 6 layered representation
        
        # Find move from squares
        squares = {}
        for action in self.data[state]:
            movefrom = chess.Move.from_uci(action).from_square
            if movefrom not in squares:
                squares[movefrom] = [self.data[state][action], 1]
            else:
                squares[movefrom][0] += self.data[state][action]
                squares[movefrom][1] += 1
        
        # Now lets take the average value for each square
        expected_output = [] # Make a 8x8 chess board representing piece picker values
        for i in range(64):
            if i in squares:
                expected_output.append(Node(squares[i][0]/squares[i][1]))
            else:
                expected_output.append(Node(0))

        return input, expected_output

    def SampleMovePickerData(self, piece: chess.PieceType) -> Tuple[Sequence[Node], Sequence[Node]]:
        """
        Sample random data from the Q-Table to be used as training data for a move picker network
        """
        found = False 
        while not found: # Repeat until we find a state that contains the piece

            state = random.choice(list(self.data)) # Select a random state
            board = chess.Board(state) # Restore the state from the FEN notation
            input = chess_utils.serialize(board) # Turn the state into 6 layered representation
            expected_output = [Node(0) for _ in range(64)] # Create an 8x8 chess board with all zeros
            
            # Update the values in the chess board with the values in our Q-Table
            # We have to store values in a temporary dictionary, since the same piece type potentially can pick the same place to moveto
            # For example two knights able to move to the same spot to check the king and snatch the queen.
            squares = {}
            for action in self.data[state]:
                if board.piece_at(move.from_square).piece_type != piece:
                    continue # Don't update for other piece types

                found = True # Found a move to match the piece, so no need to repeat for a new state

                moveto = chess.Move.from_uci(action).to_square
                if moveto not in squares:
                    squares[moveto] = (self.data[state][action], 1)
                else:
                    squares[moveto][0] += self.data[state][action]
                    squares[moveto][1] += 1
            
            # Now lets take the average value for each square
            expected_output = [] # Make a 8x8 chess board representing piece picker values
            for i in range(64):
                if i in squares:
                    expected_output.append(Node(squares[i][0]/squares[i][1]))
                else:
                    expected_output.append(Node(0))
    
        return input, expected_output