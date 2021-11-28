import numpy as np
from typing import List


class Reward:
    """ A class implemented a reward function.
    """

    def __init__(self, n_states: List[int], alpha: float) -> None:
        self.n_states = n_states
        self.alpha = alpha
        theta = np.random.rand(*n_states)
        self.theta = theta/theta.sum()

    def __call__(self) -> np.ndarray:
        return self.theta

    def update(self, grad):
        self.theta += self.alpha*grad
        self.theta /= self.theta.sum()
