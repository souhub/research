from modules.planner import PolicyIterationPlanner
from modules.irl import MaxEntIRL, MaxEntDeepIRL
from ..base import ParamsBase

import gym
import matplotlib.pyplot as plt
import numpy as np


class Params(ParamsBase):
    def __init__(self) -> None:
        super().__init__(env_name='FROZEN_LAKE_V0')
        self.n_trajectories = self.params.getint('NTrajectories')
        self.n_time_steps = self.params.getint('NTimeSteps')
        self.alpha = self.params.getfloat('Alpha')
        self.n_epoch = self.params.getint('NEpoch')
        self.is_succeeded_data_only = self.params.getboolean(
            'IsSucceededDataOnly')
        self.is_deep_learning = self.params.getboolean('IsDeepLearning')


def train(params: Params):
    env = gym.make('FrozenLake-v0')
    print("Collecting expert's trajectories in progress...")
    planner = PolicyIterationPlanner(env)
    trajectories = planner.sample_trajectories(
        params.n_trajectories, params.n_time_steps, params.is_succeeded_data_only)
    algorithm_name = "Maximum entropy deep IRL" if params.is_deep_learning else "Maximum entropy IRL"
    print('{} is running...'.format(algorithm_name))
    max_ent_irl = MaxEntDeepIRL(
        env, params.alpha, params.n_epoch) if params.is_deep_learning else MaxEntIRL(env, params.alpha, params.n_epoch)
    R = max_ent_irl.irl(trajectories)
    return R, check_trajactories_status(trajectories)


def render(R, trajectories_info: str, params: Params):
    plt.pcolor(R.reshape(4, 4)[::-1, :])
    algorithm_name = "Maximum entropy deep IRL" if params.is_deep_learning else "Maximum entropy IRL"
    plt.title("{} \n Time steps: {}, α={}, epoch={} \n {}".format(
        algorithm_name, params.n_time_steps, params.alpha, params.n_epoch, trajectories_info))
    plt.colorbar()
    plt.show()


def check_trajactories_status(trajactories: np.ndarray):
    n_failed_trajs = np.sum(trajactories == 15, axis=0)[-1]
    return "{}/{} trajectories are succeeded.".format(n_failed_trajs, len(trajactories))


def main():
    params = Params()
    R, info = train(params)
    render(R, info, params)
