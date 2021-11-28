import numpy as np
from envs import MarineBiomesEnv
from agents.rl import QLearningAgent
from agents.irl import MaxEntAgent

from utils.discretization import discrete_state
from utils.data import DataProcessor, get_trajectories
from utils.render import Render

INPUT_DATA = './input/data.csv'
N_TRAJECTORIES = 51
N_TIMESTEP = 51

N_EPOCH = 100
N_STATE_BIOMASS = 200
N_STATE_TEMPRETURE = 50

RL_ALPH = 0.2
RL_EPSILON = 0.3
RL_GAMMA = 0.9

IRL_ALPHA = 0.2

REWARD_FILE = f"reward_b{N_STATE_BIOMASS}_t{N_STATE_TEMPRETURE}_e{N_EPOCH}.txt"
POLICY_FILE = f"policy_b{N_STATE_BIOMASS}_t{N_STATE_TEMPRETURE}_e{N_EPOCH}.npy"
QTABLE_FILE = f"q_table_b{N_STATE_BIOMASS}_t{N_STATE_TEMPRETURE}_e{N_EPOCH}.npy"


def compute_reward_func() -> np.ndarray:
    env = MarineBiomesEnv()
    n_states = [N_STATE_BIOMASS, N_STATE_TEMPRETURE]
    n_action = env.action_space.n
    state_limits = [(env.min_biomas, env.max_biomass),
                    (env.min_tempreture, env.max_tempreture)]
    trajectories_info = (N_TRAJECTORIES, N_TIMESTEP)
    trajectories = get_trajectories(INPUT_DATA, n_states, state_limits)

    rl_agent = QLearningAgent(
        n_states, n_action, env, RL_ALPH, RL_EPSILON, RL_GAMMA, state_limits)
    irl_agent = MaxEntAgent(n_states, env, IRL_ALPHA,
                            trajectories, trajectories_info, state_limits)

    # Initialize a reward function by IRL.
    reward_func = irl_agent.get_reward_func()

    # epoch回ループ
    for e in range(N_EPOCH):
        # Compute the reward function from the updated policy by RL.
        rl_agent.learn(reward_func)
        policy = rl_agent.get_policy()
        # Compute a new policy from the reward function estimated by IRL.
        irl_agent.learn(policy)
        reward_func = irl_agent.get_reward_func()
        print(f'{e+1}/{N_EPOCH} times is completed.')

    dp = DataProcessor()
    dp.save_data(reward_func, f'./output/reward/{REWARD_FILE}')
    dp.save_numpy_data(rl_agent.q_table.table,
                       f'./output/q_table/{QTABLE_FILE}')
    dp.save_numpy_data(policy, f'output/policy/{POLICY_FILE}')
    return reward_func


def simulate_with_optimal_policy() -> tuple:
    env = MarineBiomesEnv()
    n_states = [N_STATE_BIOMASS, N_STATE_TEMPRETURE]
    state_limits = [(env.min_biomas, env.max_biomass),
                    (env.min_tempreture, env.max_tempreture)]

    dp = DataProcessor()
    optimal_policy = dp.read_numpy_data(f'./output/policy/{POLICY_FILE}')

    biomass = []
    tempreture = []
    biomass_variation = []

    for _ in range(100):
        done = False
        _state = env.reset()

        while not done:
            _b, _t = _state
            biomass.append(_b)
            tempreture.append(_t)

            state = discrete_state(_state, n_states, state_limits)
            b, t = state
            action = np.argmax(optimal_policy[b, t])
            _next_state, _, done, _ = env.step(action)
            _nb, _nt = _next_state
            biomass_variation.append(_nb-_b)
            _state = _next_state

    return np.array(biomass), np.array(tempreture), np.array(biomass_variation)


def render():
    dp = DataProcessor()
    reward_func = dp.read_txt(f'./output/reward/{REWARD_FILE}')
    render = Render()
    # render.imshow(reward_func)
    render.scatter3d(*simulate_with_optimal_policy())
    render.show()


render()
