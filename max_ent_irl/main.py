import gym

from planner import PolicyIterationPlanner
from irl import MaxEntIRL
from deep_irl import MaxEntDeepIRL
import matplotlib.pyplot as plt
import numpy as np

TRAJECTORIES = 200
STEPS = 20
ALPHA = 0.001
EPOCH = 200
IS_SUCCEEDED_DATA_ONLY = False
IS_DEEP_LEARNING = True


def train(n_trajectories: int, n_steps: int, alpha: float, epoch: int, is_succeeded_data_only: bool, is_deep_learning: bool):
    env = gym.make('FrozenLake-v0')
    print("Collecting expert's trajectories in progress...")
    planner = PolicyIterationPlanner(env)
    trajectories = planner.sample_trajectories(
        n_trajectories, n_steps, is_succeeded_data_only)
    algorithm_name = "Maximum entropy deep IRL" if is_deep_learning else "Maximum entropy IRL"
    print('{} is running...'.format(algorithm_name))
    max_ent_irl = MaxEntDeepIRL(
        env, alpha, epoch) if is_deep_learning else MaxEntIRL(env, alpha, epoch)
    R = max_ent_irl.irl(trajectories)
    return R, check_trajactories_status(trajectories)


def render(R, trajectories_info: str):
    plt.pcolor(R.reshape(4, 4)[::-1, :])
    algorithm_name = "Maximum entropy deep IRL" if IS_DEEP_LEARNING else "Maximum entropy IRL"
    plt.title("{} \n Time steps: {}, α={}, epoch={} \n {}".format(
        algorithm_name, STEPS, ALPHA, EPOCH, trajectories_info))
    plt.colorbar()
    plt.show()


def check_trajactories_status(trajactories: np.ndarray):
    n_failed_trajs = np.sum(trajactories == 15, axis=0)[-1]
    return "{}/{} trajectories are succeeded.".format(n_failed_trajs, len(trajactories))


if __name__ == '__main__':
    R, info = train(TRAJECTORIES, STEPS, ALPHA, EPOCH,
                    IS_SUCCEEDED_DATA_ONLY, IS_DEEP_LEARNING)
    render(R, info)
