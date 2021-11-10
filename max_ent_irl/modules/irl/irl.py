import numpy as np
import gym

from ..planner.planner import PolicyIterationPlanner
from ..q_learning.monte_carlo2 import MonteCarloAgent


class Reward:
    """A class implemented a reward function.
    """

    def __init__(self, n_features: int, alpha: float) -> None:
        self.n_features = n_features
        self.alpha = alpha
        self.theta = np.zeros((n_features))
        self.features = np.eye(n_features)

    def __call__(self):
        return np.dot(self.theta.T, self.features)

    def learn(self, grad):
        # self.theta = (1-self.alpha)*self.theta+np.exp(self.alpha*grad)
        # self.theta += np.exp(self.alpha*grad)
        self.theta += self.alpha*grad
        # Normalization
        self.theta /= self.theta.sum()


class MaxEntIRL:
    """A class implemented the maximum entropy IRL.

    Arguments:
        env: An environment of gym.
        alpha: A learning rate.
    """

    def __init__(self, env: gym.Env, alpha: float, epoch: int, rl_algorithm: str) -> None:
        self.env = env
        self.monte_carlo_agent = MonteCarloAgent(self.env, alpha)
        self.planner = PolicyIterationPlanner(self.env)
        self.alpha = alpha
        self.epoch = epoch
        self.rl_algorithm = rl_algorithm

    def irl(self, trajectories: np.ndarray):
        """An algorithm for the maximum entropy IRL.

        Compute the reward function given the expert's trajectories using the maximum entropy IRL algorithm proposed in the paper by Ziebart et al. (2008).

        Args:
            trajectories: A list of `Trajectory` instances representing the expert demonstrations.
            epoch: An integer deciding how many times to execute inversement reinforcement learning.
        """

        features = np.eye(self.env.observation_space.n)
        expert_features = self.compute_expert_features(trajectories)
        R = Reward(self.env.observation_space.n, self.alpha)

        for e in range(self.epoch):

            # Compute a new policy using the reward function.
            if self.rl_algorithm == 'planner':
                self.planner.R = R()
                V, policy = self.planner.plan()
            elif self.rl_algorithm == 'q_learning':
                policy = self.monte_carlo_agent.learn(R())

            # Compute a SVF using the policy.
            P = self.compute_svf(trajectories, policy)

            # Compute a gradient and update θ.
            grad = expert_features-np.dot(features, P)
            R.learn(grad)
        return R()

    def compute_expert_features(self, trajectories) -> np.ndarray:
        """
        """
        features = np.zeros(self.env.observation_space.n)
        for traj in trajectories:
            for s in traj:
                features[s] += 1
        features /= len(trajectories)
        return features

    def compute_svf(self, trajectories: np.ndarray, policy) -> np.ndarray:
        """
        """
        n_trajectories, n_steps = trajectories.shape
        svf = np.zeros((n_steps, self.env.observation_space.n))

        for trajectory in trajectories:
            svf[0, trajectory[0]] += 1
            svf /= n_trajectories

        for t in range(1, n_steps):
            for action in range(self.env.action_space.n):
                for state in range(self.env.observation_space.n):
                    for prob, next_state, _, _ in self.env.P[state][action]:
                        action_prob = policy[state][action]
                        svf[t][next_state] += svf[t-1][state]*action_prob*prob
        return svf.sum(axis=0)
