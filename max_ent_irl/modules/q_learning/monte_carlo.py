import gym
import numpy as np


class MonteCarloAgent():
    def __init__(self, env: gym.Env, alpha: float, epsilon=0.2) -> None:
        self.env = env
        self.gamma = 0.9
        self.alpha = alpha
        self.epsilon = epsilon
        self.experiences_log = []
        self.q_table = np.array([[0 for _ in range(env.action_space.n)]
                                 for _ in range(env.observation_space.n)])
        self.policy = {}

    # epsilon-greedy
    def take_action(self, state):
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0, self.env.action_space.n)
        else:
            action = np.argmax(self.q_table[state])
        return action

    def reset(self):
        self.experiences_log = []

    def log(self, state, action, reward):
        experience = {'state': state, 'action': action, 'reward': reward}
        self.experiences_log.append(experience)

    # def learn(self):
    #     for i, experience in enumerate(self.experiences_log):
    #         state, action, reward = experience.values()
    #         G = 0
    #         t = 0
    #         for _ in range(i, len(experience)):
    #             G += (self.gamma**t)*reward
    #             t += 1
    #         self.q_table[state][action] = (
    #             1-self.alpha)*self.q_table[state][action]+self.alpha*G
    def learn(self, R):
        self.reset()
        # state = self.env.reset()
        state = np.random.randint(0, self.env.action_space.n)
        done = False

        # Simulate to get experiences.
        while not done:
            action = self.take_action(state)
            next_state, _, done, _ = self.env.step(action, state)
            reward = R[state]
            self.log(state, action, reward)
            state = next_state

        # Learn from the experiences.
        for i, experience in enumerate(self.experiences_log):
            state, action, reward = experience.values()
            G = 0
            t = 0
            for _ in range(i, len(experience)):
                G += (self.gamma**t)*reward
                t += 1
            self.q_table[state][action] = (
                1-self.alpha)*self.q_table[state][action]+self.alpha*G

        # get policy
        self.initialize()
        for s in range(self.env.observation_space.n):
            q_values = self.q_table[s]
            best_action = np.argmax(q_values)

            for a in range(self.env.action_space.n):
                prob = 1 if a == best_action else 0
                self.policy[s][a] = prob

        return self.policy

    def initialize(self):
        """Initialize a policy.
        """
        self.policy = {}
        for s in range(self.env.observation_space.n):
            self.policy[s] = {}
            for a in range(self.env.action_space.n):
                self.policy[s][a] = 1/self.env.action_space.n

    # def simulate(self):
    #     self.reset()
    #     state = self.env.reset()
    #     done = False

    #     while not done:
    #         action = self.policy(state)
    #         next_state, reward, done, info = self.env.step(action)
    #         self.log(state, action, reward)
    #         state = next_state

    #     self.learn()
