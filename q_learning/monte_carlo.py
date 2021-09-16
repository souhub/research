import gym
import numpy as np

from utils import digitize_state


class MonteCarloAgent:
    def __init__(self, gamma, alpha) -> None:
        self.gamma = gamma
        self.alpha = alpha
        self.experiences_log = []
        self.q_table = [[0 for _ in range(3)] for _ in range(4**2)]

    def policy(self, observation=None):
        return np.random.randint(0, 3)

    def reset(self):
        self.experiences_log = []

    def log(self, state, action, reward):
        state = digitize_state(state)
        experience = {'state': state, 'action': action, 'reward': reward}
        self.experiences_log.append(experience)

    def learn(self):
        for i, experience in enumerate(self.experiences_log):
            state, action, reward = experience.values()
            G = 0
            t = 0
            for _ in range(i, len(experience)):
                G += (self.gamma**t)*reward
                t += 1
            self.q_table[state][action] = (
                1-self.alpha)*self.q_table[state][action]+self.alpha*G


if __name__ == '__main__':
    EPISODE_NUM = 10000
    TIME_NUM = 1
    GAMMA = 0.99
    ALPHA = 0.2

    agent = MonteCarloAgent(GAMMA, ALPHA)
    env = gym.make('MountainCar-v0')

    for e in range(EPISODE_NUM):
        agent.reset()
        state = env.reset()
        done = False
        # for t in range(TIME_NUM):
        while not done:
            # env.render()
            action = agent.policy(state)
            next_state, reward, done, info = env.step(action)
            agent.log(state, action, reward)
            state = next_state

        agent.learn()
        if e % 100 == 0:
            print(
                '----------------------------{}回目----------------------------------'.format(e))
            print(agent.q_table)
    env.close()
