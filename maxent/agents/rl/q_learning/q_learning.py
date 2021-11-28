import gym
import numpy as np
from typing import List

from utils.discretization import discrete_state
from ..policy import Policy
from .q_table import QTable


class QLearningAgent:
    def __init__(self, n_states: List[int], n_action: int, env: gym.Env, alpha: float, epsilon: float, gamma: float, state_limits: List[float]) -> None:
        self.n_states = n_states
        self.n_action = n_action
        self.n_feature = len(n_states)
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.state_limits = state_limits
        self.q_table = QTable(n_states, n_action, alpha, gamma)

    # epsilon-greedy
    def take_action(self, state) -> int:
        if np.random.uniform() > self.epsilon or self.q_table()[state].sum() == 0:
            action = np.random.randint(0, self.n_action)
        else:
            action = np.argmax(self.q_table()[state])
        return action

    def learn(self, reward_func: np.ndarray) -> None:
        _state = self.env.reset()
        done = False

        while not done:
            state = discrete_state(_state, self.n_states, self.state_limits)
            action = self.take_action(state)
            _next_state, _, done, _ = self.env.step(action)
            _state = _next_state
            next_state = discrete_state(
                _next_state, self.n_states, self.state_limits)
            reward = reward_func[state]
            # Update q_table
            self.q_table.update(
                state, action, next_state, reward)

    def get_policy(self) -> np.ndarray:
        policy = Policy(self.n_states, self.n_action)
        return policy(self.q_table)
