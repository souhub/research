import gym
from gym import spaces
from gym.utils import seeding
import math

import numpy as np


class MarineBiomesWithMultipleActions(gym.Env):
    """
    Description:
        The agent (a biomes which live in the sea) is started at a point where the biomass is near 100. For any given
        state the agent may choose to increase or maintain the biomass.
    Source:
        The environment is my original for the research.
    Observation:
        Type: Box(2)
        Num    Observation               Min            Max
        0      Biomass                   0              1000
        1      Tempreture                5              25
    Actions:
        Type: Discrete(2)
        Num    Action
        0      Decrease the biomass.
        1      Maintain the biomass.
        2      Increase the biomass.

        Type: Discrete(3)
        Num    Action
        0      Decrease the tempreture.
        1      Maintain the tempreture.
        2      Increase the tempreture.

    Reward:
        Reward is none, because it's decided by IRL.
    Starting State:
        The biomass of the biological model is assigned a uniform random value in
        [95, 105].
        The starting tempreture of the biological model is always assigned to 15.
    Episode Termination:
        Episode length (Timestep) is greater than 200
    """

    def __init__(self) -> None:
        self.min_biomas = 0
        self.max_biomass = 1000
        self.min_tempreture = 5
        self.max_tempreture = 25
        self.gioal_biomass = 999

        self.relative_acceleration = 0.02
        self.tempreture_response_coefficient = 0.1
        self.environmental_capacity = self.max_biomass
        self.timestep = 0
        self.tempreture_cycle = 25

        self.low = np.array(
            [self.min_biomas, self.min_tempreture], dtype=np.float32)
        self.high = np.array(
            [self.max_biomass, self.max_tempreture], dtype=np.float32)

        self.action_space_biomass = spaces.Discrete(3)
        self.action_space_tempreture = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            self.low, self.high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        biomass, tempreture = self.state
        action_biomass, action_tempreture = action

        biomass_variation = (action_biomass-1) * self.relative_acceleration * \
            math.exp(self.tempreture_response_coefficient*tempreture) * \
            (1-biomass/self.environmental_capacity)*biomass
        biomass += biomass_variation
        biomass = np.clip(biomass, self.min_biomas, self.max_biomass)

        tempreture = (action_tempreture-1) * ((self.min_tempreture+self.max_tempreture)/2) + \
            ((self.max_tempreture - self.min_tempreture)/2) * \
            math.sin((2*math.pi / self.tempreture_cycle)*self.timestep)
        tempreture = np.clip(
            tempreture, self.min_tempreture, self.max_tempreture)
        self.timestep += 1

        done = bool(self.timestep >= 200)
        reward = None

        self.state = (biomass, tempreture)
        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(self):
        self.state = np.array(
            [self.np_random.uniform(low=95, high=105), 15])
        return np.array(self.state, dtype=np.float32)
