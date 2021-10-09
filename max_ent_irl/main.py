import gym

from planner import PolicyIterationPlanner
from irl import MaxEntIRL
import matplotlib.pyplot as plt

TRAJECTORIES = 20
STEPS = 7
ALPHA = 0.1
IS_SUCCEEDED_DATA_ONLY = True


def train(n_trajectories: int, n_steps: int, alpha: float, is_succeeded_data_only: bool):
    env = gym.make('FrozenLake-v0')
    print("Collecting expert's trajectories from its demonstration...")
    planner = PolicyIterationPlanner(env)
    trajectories = planner.sample_trajectories(
        n_trajectories, n_steps, is_succeeded_data_only)

    print('Maximum entropy IRL is running...')
    max_ent_irl = MaxEntIRL(env, alpha)
    R = max_ent_irl.irl(trajectories)
    return R


def render(R):
    plt.pcolor(R.reshape(4, 4)[::-1, :])
    plt.title("Expert's tranjectories: {}, Time steps: {}".format(
        TRAJECTORIES, STEPS))
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    R = train(TRAJECTORIES, STEPS, ALPHA, IS_SUCCEEDED_DATA_ONLY)
    render(R)
