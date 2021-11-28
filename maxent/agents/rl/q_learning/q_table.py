import numpy as np
from typing import List


class QTable:
    """A class implemented Q table.
    """

    def __init__(self, n_states: List[int], n_action: int, alpha: float, gamma: float) -> None:
        self.alpha = alpha
        self.gamma = gamma
        size = tuple(n_states+[n_action])
        self.table = np.zeros(size)

    def __call__(self) -> np.ndarray:
        return self.table

    def update(self, state: tuple, action: int, next_state: tuple, reward: float) -> None:
        gain = reward+self.gamma*max(self.table[next_state])
        estimated = self.table[state][action]
        self.table[state][action] += self.alpha*(gain-estimated)
