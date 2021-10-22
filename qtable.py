import random
from typing import Dict
from chess import Move, Board
from policy import Policy

class QTable(Policy):
    def __init__(self, learning_rate = 0.1, discount_rate = 1.0, ε = 0.3) -> None:
        """
        learning_rate: Adjust how much the value is updated when new rewards are found
        discount_rate: Adjust how much the next states expected value should influence the reward
        ε: Control how often the table should be used. 0 = always, 1 = never
        """
        self.table:Dict[str, Dict[str, float]] = {}
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.ε = ε
    
    def pick_move(self, state: Board) -> Move:
        s = str(state.fen()) # Convert the board into a hashable representation that can be used to restore the board

        # Check if state is in table and roll dice to see if the look up table should be used
        if s in self.table and random.uniform(0,1) > self.ε:
            actions = self.table[s] # Lookup action
            return Move.from_uci(max(actions, key=actions.get)) # Convert from string to Move
        
        # If we don't have the state or we didn't hit a high enough number by chance, then take a random action
        return random.choice([m for m in state.legal_moves]) 

    def expected_reward(self, state: Board, action: Move) -> float:
        # Convert the board and action into hashable representations that can be restored later
        s = str(state.fen()) 
        a = action.uci() 

        # If we have the state and action, then return the value we have stored for it
        return self.table[s][a] if s in self.table and a in self.table[s] else 0

    def _max(self, state: Board) -> float:
        s = str(state.fen()) # Convert board into hashable value 
        
        # If we don't have the state, then early out
        if s not in self.table:
            return 0

        return max(self.table[s].values()) # Find the max value for all actions in state

    def update(self, state: Board, action: Move, reward: float, new_state: Board) -> None:
        # Convert the board and action into hashable representations that can be restored later
        s = str(state.fen())
        a = action.uci()

        # Make sure the state and action is in the table
        if (s not in self.table):
            self.table[s] = { }
        if (a not in self.table[s]):
            self.table[s][a] = 0

        # Update the value for state->action:
        # 1. Previously stored value
        # 2. Append reward scaled by a learning rate:
        #   2a. Take the new reward
        #   2b. Append the maximum expected reward from the resulting state adjusted with a rebate
        #   2c. Subtract the current value
        self.table[s][a] = self.table[s][a] + self.learning_rate * (reward + self.discount_rate * self._max(new_state) - self.table[s][a])