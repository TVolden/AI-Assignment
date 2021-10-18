# Created by Thomas Volden
from typing import Sequence
import chess
import numpy as np
import signal
from mcts import DecisiveMovePolicy, MonteCarloTreeSearch

if __name__ == '__main__':
    mate1_puzzle = "k7/8/8/8/8/8/2R5/1R1K4 w - - 0 1"
    mate2_puzzle = "r2qkb1r/pp2nppp/3p4/2pNN1B1/2BnP3/3P4/PPP2PPP/R2bK2R w KQkq - 1 1"
    preset = chess.Board.starting_fen

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
        print()
        print("---- Game moves ----")
        print(chess.Board(preset).variation_san(board.move_stack)) 

    mcst1 = MonteCarloTreeSearch(exploration_weight=1.5, time_budget=60, policy=DecisiveMovePolicy())
    mcst1p = MonteCarloTreeSearch(exploration_weight=1.5, time_budget=60, parallel=True, policy=DecisiveMovePolicy())
    #players = (human_input, mcst.choose)
    players = (mcst1p.choose, mcst1.choose)
    player = 1

    def handler(signum, frame):
        print_moves()
        exit(1)

    signal.signal(signal.SIGINT, handler)

    games = 1
    wins = 0
    turn_limit = 60 # 0 = no turn limit
    for n in range(games):
        player = 1
        turn = 1
        board = chess.Board(preset)
        while is_running(board) and (turn_limit == 0 or turn <= turn_limit):
            player = 1 - player
            #print("---- Player %s's turn ----" %(turn+1))
            move = players[player](board)
            print("Turn %s: Player %s moved: %s" %(turn, player+1, board.san(move)))
            board.push(move)
            #print(board)
            turn += player

        if board.is_stalemate():
            print("It's a draw!")
        elif board.is_game_over():
            print("Player %s wins!" %(player+1))
            if player == 0:
                wins += 1
        else:
            print("Max turns reached!")
        
        print_moves()

        #print("Results: %s/%s" % (wins, (n+1)))
    
    print()
    print("Statistics: %s/%s" % (wins,  games))