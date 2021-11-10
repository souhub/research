import gym
from gym import spaces


class LogisticCurve(gym.Env):
    """
    Discription:
        A biological model which follow a logistic curve and change the biomass.
        The respectively ideal values are like following, whose error range is ±2.5%.
        Initial value: 100
        Maximum relative acceleration: 0.2
        Environment capacity: 1000

    Observation:
        Type: Box(1)
        Num   Observation       Min    Max
        0     Current biomass   97.5   1025

    Actions:
        Type: Discrete(2)
        Num   Action
        0     Increase the biomass.
        1     Reduce the biomass.

    Reward:

    Starting State:

    Episode Termination:
    """

    def __init__(self) -> None:
        self.action_space = spaces.Discrete(2)

    def reset(self):
        return super().reset()

    def step(self, action):
        return super().step(action)

    def observe(self):
        return
