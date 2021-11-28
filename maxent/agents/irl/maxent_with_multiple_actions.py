import numpy as np
import gym
from typing import List
import itertools

from utils.discretization import discrete_state
from .reward import Reward

class MaxEntWithMultipleActionsAgent:
    def __init__(self, n_states: List[int], n_actions: List[int], env: gym.Env, alpha: float, trajectories: np.ndarray, trajectories_info, state_limits) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_feature = len(n_states)
        self.state_limits = state_limits
        self.env = env
        self.alpha = alpha
        self.trajectories = trajectories
        self.trajectories_info = trajectories_info

        self.reward_funcs = [Reward(n_states[n], alpha)
                             for n in range(self.n_feature)]

    def learn(self, policies):
        features = [np.eye(self.n_states[n])
                    for n in range(self.n_feature)]
        expert_features = self.__compute_expert_features()
        svf = self.__compute_svf(policies)
        for n in range(self.n_feature):
            grad = expert_features[n]-np.dot(features[n], svf[n])
            self.reward_funcs[n].update(grad)

    def get_reward_funcs(self) -> List[Reward]:
        return [self.reward_funcs[n]() for n in range(self.n_feature)]

    def __compute_expert_features(self):
        n_trajectories, n_steps = self.trajectories_info
        expert_features = [np.zeros(self.n_states[n])
                           for n in range(self.n_feature)]
        for n in range(self.n_feature):
            for trajectory in self.trajectories:
                traj = trajectory[n]
                for s in traj:
                    expert_features[n][s] += 1
            expert_features[n] /= n_trajectories
        print('expert_features', expert_features)
        return expert_features

    def __compute_svf(self, policies) -> np.ndarray:
        n_trajectories, n_steps = self.trajectories_info
        features = [np.zeros((n_steps, self.n_states[n]))
                    for n in range(self.n_feature)]

        for n in range(self.n_feature):
            for trajectory in self.trajectories:
                traj = trajectory[n]
                features[n][0, traj[0]] += 1
            features[n] /= n_trajectories
        actions = list(itertools.product(
            *[tuple(range(self.n_actions[n])) for n in range(self.n_feature)]))
        for n in range(self.n_feature):
            for t in range(1, n_steps):
                # for action in range(self.n_actions[n]):
                for action in actions:
                    for state in range(self.n_states[n]):
                        _next_state, _, _, _ = self.env.step(action)
                        next_state = discrete_state(
                            _next_state, self.n_states, self.state_limits)
                        action_prob = policies[n][state][action[n]]
                        features[n][t][next_state[n]] += \
                            features[n][t - 1][state]*action_prob
        return [features[n].sum(axis=0) for n in range(self.n_feature)]
