import chess

class Policy:
    def pick_move(self, board: chess.Board) -> chess.Move:
        pass

    def expected_reward(self, board: chess.Board, action: chess.Move) -> float:
        return 0

    def update(self, state: chess.Board, action: chess.Move, reward: float, new_state: chess.Board) -> None:
        pass