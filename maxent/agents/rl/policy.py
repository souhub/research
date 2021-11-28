import numpy as np
from typing import List
import itertools


class Policy:
    """A class implemented a policy"""

    def __init__(self, n_states: List[int], n_action: int) -> None:
        self.n_states = n_states
        self.n_feature = len(n_states)
        self.n_action = n_action
        self.size = tuple(n_states+[n_action])

    def __call__(self, q_table) -> np.ndarray:
        policy = np.full(self.size, 1/self.n_action)
        states = itertools.product(
            *tuple(list(range(n_state)) for n_state in self.n_states))

        for state in states:
            q_values = q_table()[state]
            best_action = np.argmax(q_values)
            for a in range(self.n_action):
                prob = 1 if a == best_action else 0
                policy[state][a] = prob
        return policy
