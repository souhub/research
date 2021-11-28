import numpy as np
import gym
from typing import List
import itertools

from utils.discretization import discrete_state
from .reward import Reward


class MaxEntAgent:
    """A class implemented the maximum entropy IRL.

    Arguments:
        n_states: List of the discrete number per feature value.
        env: An environment of gym.
        alpha: A learning rate.
        trajectories: Expert trajectoreies.
    """

    def __init__(self,  n_states: List[int], env: gym.Env, alpha: float, trajectories: np.ndarray, trajectories_info, state_limits) -> None:
        self.n_states = n_states
        self.n_action = env.action_space.n
        self.n_feature = len(n_states)
        self.state_limits = state_limits
        self.env = env
        self.alpha = alpha
        self.trajectories = trajectories
        self.trajectories_info = trajectories_info
        self.reward_func = Reward(n_states, alpha)

    def learn(self, policy):
        expert_features = self.__compute_expert_features()
        svf = self.__compute_svf(policy)
        grad = expert_features-svf
        self.reward_func.update(grad)

    def get_reward_func(self) -> np.ndarray:
        return self.reward_func()

    def __compute_expert_features(self):
        n_trajectories, n_steps = self.trajectories_info
        expert_features = np.zeros(self.n_states)
        for trajectory in self.trajectories:
            for t in range(n_steps):
                state = tuple([trajectory[n][t]
                               for n in range(self.n_feature)])
                expert_features[state] += 1
        expert_features /= n_trajectories
        return expert_features

    def __compute_svf(self, policy) -> np.ndarray:
        n_trajectories, n_steps = self.trajectories_info
        features = np.ones([n_steps]+self.n_states)

        for trajectory in self.trajectories:
            features[0, trajectory[0]] += 1
        features /= n_trajectories

        states = itertools.product(
            *tuple(list(range(n_state)) for n_state in self.n_states))
        for t in range(1, n_steps):
            for action in range(self.n_action):
                for state in states:
                    _next_state, _, _, _ = self.env.step(action)
                    next_state = discrete_state(
                        _next_state, self.n_states, self.state_limits)
                    action_prob = policy[state][action]
                    features[t][next_state] += \
                        features[t - 1][state]*action_prob
        return features.sum(axis=0)
