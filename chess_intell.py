import random
import chess
import numpy as np
from typing import Sequence
from numpy.lib.function_base import average
from chess_utils import PIECE_ORDER, serialize
from conv import Conv
from data_sampler import DataSampler
from denselayer import DenseLayer
from mln import MultiLayeredNetwork
from nn_layer import NNLayer
from node import Node
from policy import Policy
from qtable import QTable
from chess import *
from sgd import SGD
import tqdm

def ReLU(x:Node) -> Node:
    return x.relu()

def sum_nodes(nodes:Sequence[Node]) -> Node:
    val = Node(0)
    for node in nodes:
        val += node
    return val

def max_index(nodes:Sequence[Node]) -> int:
    idx = 0
    max = float('-inf')
    for i, node in enumerate(nodes):
        if node.value > max:
            max = node.value
            idx = i
    return idx

class SoftMax(NNLayer):
    def forward(self, inputs: Sequence[Node]) -> Sequence[Node]:
        e = np.exp(inputs)
        return e / sum_nodes(e)

class Flatten(NNLayer):
    def __init__(self, n_channels:int) -> None:
        super().__init__()
        self.channels = n_channels

    def forward(self, inputs: Sequence[Node]) -> Sequence[Node]:
        segment = len(inputs) / self.channels
        outputs = []
        for i in range(self.channels):
            val = sum_nodes(inputs[int(segment*i):int(segment*(i+1))])
            outputs.append(val/Node(segment))
        return outputs

class ChessIntell(Policy):
    def __init__(self, decoratee:Policy = QTable()) -> None:
        self.piecePicker = self._createNetwork()
        self.movePicker = [self._createNetwork() for _ in range(6)]
        self.decoratee = decoratee
    
    def _createNetwork(self) -> MultiLayeredNetwork:
        """
        Create the network architecture for both piece and move predictor.

        The architecture is my interpretation of the architecture described in 
        Barak Oshri and Nishith Khandwala paper: "Predicting Moves in Chess using Convolutional Neural Networks"
        http://cs231n.stanford.edu/reports/2015/pdfs/ConvChess.pdf
        """
        NN = [  
            Conv(32, 3, ReLU, (8,8,6)), # Take in chess board with 6 channels, one for each piece
            Flatten(32), # Flatten the convolutional network output to an average of it's channels
            DenseLayer(32, 128, ReLU), # Pass the flatten output as input for 32 nodes to 128 outputs
            DenseLayer(128, 64, ReLU), # pass the 128 outpus as input to 64 outpus
            SoftMax() # Parse the output through a softmax to get probability output
            ]
        return MultiLayeredNetwork(NN)

    def pick_move(self, board: Board) -> Move:
        # Pass the board state to get a position of a piece that the CNN recommends moving
        s_board = serialize(board)
        movefrom = max_index(self.piecePicker.forward(s_board))

        piece = board.piece_at(movefrom) # Find the on the position
        
        if piece is None: # If the piece isn't on the board, then pass the torch
            #print(f"No legal piece at {square_name(movefrom)}!")
            return self.decoratee.pick_move(board)

        # Pass the board state to get the preferred position of said piece
        cnn_index = PIECE_ORDER.index(piece.piece_type)
        moveto = max_index(self.movePicker[cnn_index].forward(s_board))
        move = Move(movefrom, moveto) # parse the information into a move
        
        if move not in board.legal_moves: # Check if the move is actually legal if not then pass the torch to the next policy
            #print(f"{move} is an illegal move!")
            return self.decoratee.pick_move(board)

        return move # Return the move

    def update(self, state: Board, action: Move, reward: float, new_state: Board) -> None:
        self.decoratee.update(state, action, reward, new_state)

    def train_piece_picker(self, sampler: DataSampler):
        # Train the piece picker network
        training_data = [sampler.SamplePiecePickerData() for _ in range(100)]
        learning_rate = 0.01
        optimizer = SGD(self.piecePicker.parameters(), learning_rate) # Use gradient descend to optimize weights
        batch = 64 # The number of training data to include in each run
        #losses = [] # The accumulated loss
        for _ in tqdm.tqdm(range(100)):
            loss = [] # Reset loss
            for _ in range(batch): # Gather results from the network
                x, y_target = random.choice(training_data) # Take random sample to avoid sorted bias
                y = self.piecePicker.forward(x) # Get the output from the network

                loss = [((y_target[i]-y[i])**2) for i in range(len(y))] # Calculate loss for each output square it
            loss = [l/Node(batch) for l in loss] # Get the mean squared loss for the whole batch
            
            #losses.append(loss) # Append the losses to the statistics
            optimizer.zero_grad() # Reset the gradient to avoid concatenated gradients

            for l in loss:
                l.backward() # Calculate the gradient using the chain rule

            optimizer.step() # Adjust weights to minimize loss using the gradient

    def train_move_picker(self, sampler: DataSampler, piece: chess.PieceType):
        # Train the piece picker network
        training_data = [sampler.SampleMovePickerData(piece) for _ in range(100)]
        learning_rate = 0.01
        cnn_index = PIECE_ORDER.index(piece.piece_type)
        optimizer = SGD(self.movePicker[cnn_index].parameters(), learning_rate) # Use gradient descend to optimize weights
        batch = 64 # The number of training data to include in each run
        #losses = [] # The accumulated loss
        for _ in tqdm.tqdm(range(100)):
            loss = [] # Reset loss
            for _ in range(batch): # Gather results from the network
                x, y_target = random.choice(training_data) # Take random sample to avoid sorted bias
                y = self.movePicker[cnn_index].forward(x) # Get the output from the network
                loss = [((y_target[i]-y[i])**2) for i in range(len(y))] # Calculate loss for each output
            loss = [l/Node(batch) for l in loss] # Get the average loss for the whole batch
            
            #losses.append(loss) # Append the losses to the statistics
            optimizer.zero_grad() # Reset the gradient to avoid concatenated gradients

            for l in loss:
                l.backward() # Calculate the gradient using the chain rule

            optimizer.step() # Adjust weights to minimize loss using the gradient