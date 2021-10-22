import chess
from mcts import DecisiveMovePolicy, MonteCarloTreeSearch

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

def mate_in_one_MCTS():
    mcst1 = MonteCarloTreeSearch(exploration_weight=1.5, time_budget=5, policy=DecisiveMovePolicy())
    

def mate_in_two_MCTS_QLearning():
    pass

def mate_in_two_MCTS_DQN():
    pass

def play_chess():
    pass