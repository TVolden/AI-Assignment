# Created by Thomas Volden
from typing import Sequence
import chess
import numpy as np
import signal
from chess_intell import ChessIntell
from chess_utils import PIECE_ORDER
from data_sampler import DataSampler
from mcts import DecisiveMovePolicy, MonteCarloTreeSearch, RandomPolicy
from policy import Policy
from qtable import QTable
import json
import time

if __name__ == '__main__':
    mate1_puzzle = "k7/8/8/8/8/8/2R5/1R1K4 w - - 0 1"
    mate2_puzzle = "r2qkb1r/pp2nppp/3p4/2pNN1B1/2BnP3/3P4/PPP2PPP/R2bK2R w KQkq - 1 1"
    preset = mate2_puzzle # chess.Board.starting_fen

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

    board = None

    def print_moves():
        print(chess.Board(preset).variation_san(board.move_stack)) 

    mcst1 = MonteCarloTreeSearch(exploration_weight=1.5, time_budget=5, policy=DecisiveMovePolicy())
#    mcst1p = MonteCarloTreeSearch(exploration_weight=1.5, time_budget=6, parallel=True, policy=DecisiveMovePolicy())
    
    # Handle a CTRL+C event from the user, make sure the current moves are printed.
    def handler(signum, frame):
        print_moves()
        exit(1)
    signal.signal(signal.SIGINT, handler)

    policy = ChessIntell(RandomPolicy())#QTable(ε=1.0))

    #with open('pCNN-params.json') as json_file:
    #    data = json.load(json_file)
    #    for i, n in enumerate(policy.piecePicker.parameters()):
    #        n.value = data[i]

    data = None
    with open('qtable.json') as json_file:
        #policy.table = json.load(json_file)
        data = json.load(json_file)

    sampler = DataSampler(data)

    # Training time!
    print("Begin training!")
    start_time = time.time() 
    policy.train_piece_picker(sampler)
    with open('pCNN-params-1.json', 'w') as outfile:
        json.dump([p.value for p in policy.piecePicker.parameters()], fp=outfile)
    print("Piece trained!")

    for p in PIECE_ORDER:
        policy.train_move_picker(sampler, p)
    print("Training ended after %s sec" % ((time.time() - start_time)))
    
    for i,p in enumerate(PIECE_ORDER):
        with open(f"mCNN{i}-params-1.json", 'w') as outfile:
            json.dump([p.value for p in policy.movePicker[i].parameters()], fp=outfile)

    print("--- Training complete, starting test! ---")

    min_epsilon = 0.1
    max_epsilon = 1.0
    decay_rate = 0.1
    episodes = 1
    exploration = 0.3
    rewards = []

    for episode in range(episodes):
        #policy.ε = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        #print(f"### ε: {policy.ε}")

        mcst1p = MonteCarloTreeSearch(exploration_weight=exploration, time_budget=60, parallel=True, policy=policy)
        #######################
        # Game configurations #
        #######################
        #players = (human_input, mcst.choose) # Play against the computer
        players = (mcst1p.choose, mcst1.choose) # Self play
        games = 10 # Set a number of matches, good for puzzle tests and self-play
        turn_limit = 2 # 0 = no turn limit
        
        wins = [0, 0] # Used for statistics, how many times have player one won
        for n in range(games):
            player = 1 # Reset the current player
            turn = 1 # Reset the number of full turns
            board = chess.Board(preset) # Reset the board state

            # Run the game until it's game over or the maximal number of full turns have been reached
            while is_running(board) and (turn_limit == 0 or turn <= turn_limit):
                player = 1 - player # Switch the player
                
                #print("---- Player %s's turn ----" %(turn+1))
                move = players[player](board) # Call the input method set in game configuration
                print(f"Turn {turn}: Player {player+1} moved: {board.san(move)}")
                board.push(move) # Alter the game state with the chosen action
                #print(board)
                turn += player # Increment the turn counter when player 2 has played

            # Check win conditions
            if board.is_stalemate():
                pass#print("It's a draw!")
            elif board.is_game_over():
                #print(f"Player {player + 1} wins!")
                wins[player] += 1 # Update the game statistics
            #else:
            #    print("Max turns reached!")
            
            print_moves()

            #print("Results: White: {1}/{0}, Black: {2}/{0}".format((n+1), wins[0], wins[1]))
        
        print()
        print(f">>> {episode+1}: White: {wins[0]}/{games}, Black: {wins[1]}/{games}")
        rewards.append(wins[0])

        #with open('qtable.json', 'w') as outfile:
        #    json.dump(policy.table, fp=outfile)

    print("Rewards over episodes:")
    print(rewards)
    