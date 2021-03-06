import chess
from chess_intell import ChessIntell
from chess_utils import PIECE_ORDER
from data_sampler import DataSampler
from mcts import DecisiveMovePolicy, MonteCarloTreeSearch
from policy import Policy
from qtable import QTable
import numpy as np
import json
import time

mate1_puzzle = "k7/8/8/8/8/8/2R5/1R1K4 w - - 0 1"
mate2_puzzle = "r2qkb1r/pp2nppp/3p4/2pNN1B1/2BnP3/3P4/PPP2PPP/R2bK2R w KQkq - 1 1"

def print_moves(initial_possition, board:chess.Board):
        print(chess.Board(initial_possition).variation_san(board.move_stack)) 

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

def e1_mate_in_one_MCTS():
    wanna_play = True
    while wanna_play:
        print("Before we begin, we have to set the parameters for the experiment.")

        p = input("Should I use parallel processes? [Yes/no] (default=yes): ")
        parallel = p.lower() not in ["no", "n"]
            
        ##############
        # Game setup #
        ##############    

        # Random policy where the legal move generator checks if an action causes game over and returns it
        policy = DecisiveMovePolicy()
        time_budget = 1
        c = 0.3
        mcts = MonteCarloTreeSearch(exploration_weight=0.3, time_budget=time_budget, parallel=parallel, policy=policy)
        players = (mcts.choose, mcts.choose) # Play against it self

        # We set turn limit to one, since the MCTS should be able to defeat the opponent in one move
        turn_limit = 1 # 0 = no turn limit

        #################
        # Initiate game #
        #################

        player = 1 # Reset the current player
        turn = 1 # Reset the number of full turns
        board = chess.Board(mate1_puzzle) # Reset the board state

        # Run the game until it's game over or the maximal number of full turns have been reached
        while not board.is_game_over() and (turn_limit == 0 or turn <= turn_limit):
            player = 1 - player # Switch the player
            
            move = players[player](board) # Call the input method set in game configuration
            print(f"Turn {turn}: Player {player+1} moved: {board.san(move)}")
            board.push(move) # Alter the game state with the chosen action
            turn += player # Increment the turn counter when player 2 has played

        print_moves(mate1_puzzle, board)

        # Check win conditions
        if board.is_stalemate():
            print("It's a draw!")
        elif board.is_game_over():
            print(f"Player {player + 1} wins!")
        else:
            print("Max turns reached!")

        wanna_play = input("Wanna try again? [Yes/No] (default=No): ").lower() in ["yes", "y"]

def e2_mate_in_two_MCTS_QLearning():
    print("Before we begin, we have to set the parameters for the experiment.")
    time_budget = 10
    tb = input(f"Time budget to train [seconds] (default={time_budget}): ")
    if tb.isnumeric() and int(tb) > 0:
        time_budget = int(tb)
    
    episodes = 10
    ep = input(f"Number of episodes: (default={episodes}): ")
    if ep.isnumeric() and int(ep) > 0:
        episodes = int(ep)

    attempts = 5
    a = input(f"Number of attempts per episode: (default={attempts}): ")
    if a.isnumeric() and int(a) > 0:
        attempts = int(a)

    policy = QTable(??=1.0)

    min_epsilon = 0.1
    max_epsilon = 1.0
    decay_rate = 0.1
    exploration = 0.3
    rewards = []

    print("## Train Q-Table ##")
    for episode in range(episodes):
        policy.?? = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        print(f"### ??: {policy.??}")

        # Setup a MCTS from experiment1 to play the opponent
        opponent = MonteCarloTreeSearch(exploration_weight=exploration, time_budget=1, policy=DecisiveMovePolicy())
        mcts_q = MonteCarloTreeSearch(exploration_weight=exploration, time_budget=5, parallel=True, policy=policy)
        
        #######################
        # Game configurations #
        #######################
        players = (mcts_q.choose, opponent.choose) # Self play
        games = attempts # Set number of attempts
        # Mate in two puzzle, should be done after two full turns
        turn_limit = 2 # 0 = no turn limit
        
        wins = [0, 0] # Used for statistics, how many times have player one won
        for n in range(games):
            player = 1 # Reset the current player
            turn = 1 # Reset the number of full turns
            board = chess.Board(mate2_puzzle) # Reset the board state

            # Run the game until it's game over or the maximal number of full turns have been reached
            while not board.is_game_over() and (turn_limit == 0 or turn <= turn_limit):
                player = 1 - player # Switch the player

                move = players[player](board) # Call the input method set in game configuration
                print(f"Turn {turn}: Player {player+1} moved: {board.san(move)}")
                board.push(move) # Alter the game state with the chosen action
                
                turn += player # Increment the turn counter when player 2 has played

            # Check win conditions
            if board.is_game_over() and not board.is_stalemate():
                wins[player] += 1 # Update the game statistics
            
            print_moves(mate2_puzzle, board)

            print("Result: White: {1}/{0}, Black: {2}/{0}".format((n+1), wins[0], wins[1]))
        
        print()
        print(f">>> {episode+1}: White: {wins[0]}/{games}, Black: {wins[1]}/{games}")
        rewards.append(wins[0])

    print("Rewards over episodes:")
    print(rewards)

    print("## Test Q-Table ##")
    policy.?? = 0.0 # Set epsilon to always use table lookup
    time_budget = 1 # Set the time budget to one second to verify an improvement
    print("Time budget set to 1 second and epsilon set to 0.")

    # Setup a MCTS from experiment1 to play the opponent
    opponent = MonteCarloTreeSearch(exploration_weight=0.3, time_budget=1, policy=DecisiveMovePolicy())
    mcts_q = MonteCarloTreeSearch(exploration_weight=exploration, time_budget=time_budget, parallel=True, policy=policy)
    
    #######################
    # Game configurations #
    #######################
    players = (mcts_q.choose, opponent.choose) # Self play
    games = attempts # Set number of attempts
    # Mate in two puzzle, should be done after two full turns
    turn_limit = 2 # 0 = no turn limit
    
    wins = [0, 0] # Used for statistics, how many times have player one won
    for n in range(games):
        player = 1 # Reset the current player
        turn = 1 # Reset the number of full turns
        board = chess.Board(mate2_puzzle) # Reset the board state

        # Run the game until it's game over or the maximal number of full turns have been reached
        while not board.is_game_over() and (turn_limit == 0 or turn <= turn_limit):
            player = 1 - player # Switch the player

            move = players[player](board) # Call the input method set in game configuration
            print(f"Turn {turn}: Player {player+1} moved: {board.san(move)}")
            board.push(move) # Alter the game state with the chosen action
            
            turn += player # Increment the turn counter when player 2 has played

        # Check win conditions
        if board.is_game_over() and not board.is_stalemate():
            wins[player] += 1 # Update the game statistics
        
        print_moves(mate2_puzzle, board)

        print("Result: White: {1}/{0}, Black: {2}/{0}".format((n+1), wins[0], wins[1]))
    
    print()
    print(f">>> Test result: White: {wins[0]}/{games}, Black: {wins[1]}/{games}")

    save = input("Want to save Q-Table? [Yes/No] (default=No): ")
    if save.lower() in ["yes", "y"]:
        with open('qtable-experimet2.json', 'w') as outfile:
            json.dump(policy.table, fp=outfile)

        print("Q-Table saved as qtable-experiment2.json")

def e3_mate_in_two_MCTS_DQN():
    # Disclaimer: the backpropagation takes for ever. 
    print("Warning! This experiment take a long time! Like half a year or so. :(")
    print("I recommend running this with a debugger and halting the process to see what's happening.")
    con = input("Continue? [Yes/No] (default=Yes): ")
    if con.lower() in ["no", "n"]:
        return

    print("Before we begin, we have to set the parameters for the experiment.")
    attempts = 5
    a = input(f"Number of attempts per episode: (default={attempts}): ")
    if a.isnumeric() and int(a) > 0:
        attempts = int(a)

    policy = ChessIntell(QTable(??=1.0))
    
    # Load data from previous Q-Table training
    data = None
    with open('qtable.json') as json_file:
        #policy.table = json.load(json_file)
        data = json.load(json_file)
    
    # Feed the data to the data sampler
    sampler = DataSampler(data)

        # Training time!
    print("Begin training!")
    start_time = time.time() # Save the start time

    # Call the trainer for the piece picker
    policy.train_piece_picker(sampler)

    # Save the weights for the piece picker
    with open('pCNN-params-1.json', 'w') as outfile:
        json.dump([p.value for p in policy.piecePicker.parameters()], fp=outfile)
    print("Piece trained!")

    # For each piece type, call the trainer for each of the 6 movement pickers
    for p in PIECE_ORDER:
        policy.train_move_picker(sampler, p)
    print("Training ended after %s sec" % ((time.time() - start_time)))
    
    for i,p in enumerate(PIECE_ORDER):
        with open(f"mCNN{i}-params-1.json", 'w') as outfile:
            json.dump([p.value for p in policy.movePicker[i].parameters()], fp=outfile)

    print("--- Training complete, starting test! ---")

    print("## Test MCTS with DQN assistance ##")
    time_budget = 1 # Set the time budget to one second to verify an improvement
    print("Time budget set to 1 second and epsilon set to 0.")

    # Setup a MCTS from experiment1 to play the opponent
    opponent = MonteCarloTreeSearch(exploration_weight=0.3, time_budget=1, policy=DecisiveMovePolicy())
    mcts_q = MonteCarloTreeSearch(exploration_weight=0.3, time_budget=time_budget, parallel=True, policy=policy)
    
    #######################
    # Game configurations #
    #######################
    players = (mcts_q.choose, opponent.choose) # Self play
    games = attempts # Set number of attempts
    # Mate in two puzzle, should be done after two full turns
    turn_limit = 2 # 0 = no turn limit
    
    wins = [0, 0] # Used for statistics, how many times have player one won
    for n in range(games):
        player = 1 # Reset the current player
        turn = 1 # Reset the number of full turns
        board = chess.Board(mate2_puzzle) # Reset the board state

        # Run the game until it's game over or the maximal number of full turns have been reached
        while not board.is_game_over() and (turn_limit == 0 or turn <= turn_limit):
            player = 1 - player # Switch the player

            move = players[player](board) # Call the input method set in game configuration
            print(f"Turn {turn}: Player {player+1} moved: {board.san(move)}")
            board.push(move) # Alter the game state with the chosen action
            
            turn += player # Increment the turn counter when player 2 has played

        # Check win conditions
        if board.is_game_over() and not board.is_stalemate():
            wins[player] += 1 # Update the game statistics
        
        print_moves(mate2_puzzle, board)

        print("Result: White: {1}/{0}, Black: {2}/{0}".format((n+1), wins[0], wins[1]))
    
    print()
    print(f">>> Test result: White: {wins[0]}/{games}, Black: {wins[1]}/{games}")

    save = input("Want to save Q-Table? [Yes/No] (default=No): ")
    if save.lower() in ["yes", "y"]:
        with open('qtable-experimet2.json', 'w') as outfile:
            json.dump(policy.table, fp=outfile)

        print("Q-Table saved as qtable-experiment2.json")

def play_chess():
    print("Before we begin, we have to set the parameters for the experiment.")
    ##############
    # Game setup #
    ##############
    policy = DecisiveMovePolicy()
    q = input("May the Monte Carlo Tree search use QTable lookups? [Yes/No] (default=No): ")
    if q.lower() in ["yes", "y"]:
        policy = QTable(??=0.3)
        with open('qtable.json') as json_file:
            policy.table = json.load(json_file)

    time_budget = 30
    tb = input(f"Time budget to train [seconds] (default={time_budget}): ")
    if tb.isnumeric() and int(tb) > 0:
        time_budget = int(tb)
   
    c = 0.3
    mcts = MonteCarloTreeSearch(exploration_weight=c, time_budget=time_budget, parallel=True, policy=policy)

    players = (human_input, mcts.choose) # Human plays white
    w = input("Which color would you like to play? [White/Black] (default=White): ")
    if w.lower() in ["black", "b"]:
        players = (mcts.choose, human_input) # Human plays black

    #################
    # Initiate game #
    #################

    player = 1 # Reset the current player
    turn = 1 # Reset the number of full turns
    board = chess.Board(chess.Board.starting_fen) # Reset the board state

    # Run the game until it's game over or the maximal number of full turns have been reached
    while not board.is_game_over():
        player = 1 - player # Switch the player
        
        move = players[player](board) # Call the input method set in game configuration
        print(f"Turn {turn}: Player {player+1} moved: {board.san(move)}")
        board.push(move) # Alter the game state with the chosen action
        print(board)
        turn += player # Increment the turn counter when player 2 has played

    print_moves(chess.Board.starting_fen, board)

    # Check win conditions
    if board.is_stalemate():
        print("It's a draw!")    
    elif board.is_game_over():
        print(f"Player {player + 1} wins!")

experiment_list = [
    ("Solve a mate in one puzzle with a MCTS algorithm.", e1_mate_in_one_MCTS),
    ("Solve a mate in two puzzle with a MCTS asssisted by Q-Learning.", e2_mate_in_two_MCTS_QLearning),
    ("Train and solve a mate in two puzzle using a MCTS assisted by DQN (takes months to run)", e3_mate_in_two_MCTS_DQN),
    ("Play chess against a computer", play_chess)
]