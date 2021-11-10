import gym
import numpy as np


class Policy:
    """A class implemented a policy
    """

    def __init__(self, env: gym.Env) -> None:
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n

    def __call__(self, q_table):
        policy = np.array([[a for a in range(self.n_actions)]
                           for _ in range(self.n_states)])
        for s in range(self.n_states):
            q_values = q_table[s]
            best_action = np.argmax(q_values)
            for a in range(self.n_actions):
                prob = 1 if a == best_action else 0
                policy[s][a] = prob
        return policy


class MonteCarloAgent():
    def __init__(self, env: gym.Env, alpha: float, epsilon=0.2) -> None:
        self.env = env
        self.gamma = 0.9
        self.alpha = alpha
        self.epsilon = epsilon
        self.experiences_log = []
        self.q_table = np.array([[0 for _ in range(env.action_space.n)]
                                 for _ in range(env.observation_space.n)])

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

    def learn(self, R):
        self.reset()
        state = self.env.reset()
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
        policy = Policy(self.env)
        return policy(self.q_table)
