from modules.irl import MaxEntIRL, MaxEntDeepIRL
from ..base import ParamsBase
from .data import DataProcessor

import gym
from gym import spaces
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
import matplotlib.colors as colors
from datetime import datetime
import os


class Params(ParamsBase):
    def __init__(self) -> None:
        super().__init__(env_name='LOGISTIC_CURVE')
        self.data_path = self.params['DataPath']
        self.n_digitize = self.params.getint('NDigitize')
        self.n_timestep = self.params.getint('NTimeStep')
        self.alpha = self.params.getfloat('Alpha')
        self.n_epoch = self.params.getint('NEpoch')
        self.is_deep_learning = self.params.getboolean('IsDeepLearning')
        self.desired_prob = 1/self.n_digitize
        self.color_thme = self.params['ColorTheme']
        self.maximum_relative_velocity = self.params.getfloat(
            'MaximumRelativeVelocity')
        self.use_existing_reward = self.params.getboolean('UseExistingReward')

        self.model_type = self.params['ModelType']
        self.rl_algorithm = self.params['RLAlgorithm']
        save_reward_filename = self.params['SaveRewardFilename']
        read_reward_filename = self.params['ReadRewardFilename']
        reward_dir = f"output/reward/{self.model_type}_model/{self.rl_algorithm}"
        self.save_reward_path = f"{reward_dir}/{save_reward_filename}"
        self.read_reward_path = f"{reward_dir}/{read_reward_filename}"


class LogisticCurve(gym.Env):
    def __init__(self, params: Params) -> None:
        self.observation_space = spaces.Discrete(params.n_digitize)
        self.action_space = spaces.Discrete(
            int(self.observation_space.n*params.maximum_relative_velocity)*2)
        self.goal_state = params.n_digitize-1
        self.params = params
        self.P = self.init_P()

    def init_P(self):
        P = {}
        for s in range(self.observation_space.n):
            ps = {}
            for a in range(self.action_space.n):
                pa = []
                desired_prob = self.params.desired_prob
                desired_next_s = s + a if a <= self.action_space.n / \
                    2 else s-(a-self.action_space.n/2)
                next_states = list(range(int(s-self.action_space.n/2),
                                         int(s+self.action_space.n/2)+1))
                for next_s in range(self.observation_space.n):
                    next_state = next_s
                    prob = (1-desired_prob) / \
                        len(next_states) if next_state in next_states else 0
                    done = bool(
                        self.goal_state <= next_state or next_state < 0)
                    reward = None
                    if next_state >= self.goal_state:
                        next_state -= 1
                    elif next_state < 0:
                        next_state = 0
                    if next_state == desired_next_s:
                        prob = desired_prob
                    pa.append((prob, next_state, reward, done))
                ps[a] = pa
            P[s] = ps
        return P

    def _reset(self):
        return np.random.randint(0, self.action_space.n)


class SimpleLogisticCurve(gym.Env):
    def __init__(self, params: Params) -> None:
        self.observation_space = spaces.Discrete(params.n_digitize)
        self.action_space = spaces.Discrete(3)
        self.goal_state = params.n_digitize-1
        self.params = params
        self.P = self.init_P()

    def init_P(self):
        P = {}
        for s in range(self.observation_space.n):
            ps = {}
            for a in range(self.action_space.n):
                pa = []

                for next_s in range(self.observation_space.n):
                    next_state = next_s
                    prob = 1/self.action_space.n if next_state == s or next_state == s - \
                        1 or next_state == s+1 else 0
                    done = bool(
                        self.goal_state <= next_state or next_state < 0)
                    reward = None
                    if next_state >= self.goal_state:
                        next_state -= 1
                    elif next_state < 0:
                        next_state = 0

                    pa.append((prob, next_state, reward, done))
                ps[a] = pa
            P[s] = ps
        return P

    def reset(self):
        return np.random.randint(0, self.action_space.n)

    def step(self, action, state):
        prob = 1/self.action_space.n
        next_state = np.random.randint(state-1, state+2)
        done = bool(
            self.goal_state <= next_state or next_state < 0)
        if done:
            next_state = state
        info = None
        return next_state, prob, done, info


def main():
    params = Params()

    # Condition branch with model_type
    env = SimpleLogisticCurve(
        params) if params.model_type == 'simple' else LogisticCurve(params)

    dp = DataProcessor()
    read_data = dp.read_data(params.data_path)
    trajectories = dp.digitize(data=read_data, n_digitize=params.n_digitize)
    if params.use_existing_reward:
        print('Use existing reward.')
        R = np.loadtxt(params.read_reward_path)
    else:
        R = calculate_R(params, trajectories, env)
        dp.save_data_to_csv(R, save_path=params.save_reward_path)
    render = Render(R, params)
    render.overview()
    render.per_time()
    render.differential_overview()
    plt.show()


def calculate_R(params: Params, trajectories, env):
    R = []
    N_TIMESTEP = params.n_timestep
    for t in range(params.n_timestep):
        print('{}/{} timestep is running...'.format(t+1, N_TIMESTEP))
        curr_trajectories = trajectories[:, t:t+1]
        R_t = MaxEntDeepIRL(env, params.alpha, params.n_epoch).irl(
            curr_trajectories) if params.is_deep_learning else MaxEntIRL(env, params.alpha, params.n_epoch, params.rl_algorithm).irl(curr_trajectories)
        R.append(R_t)
    R = np.vstack(R)
    R = R.T  # Convert T✕S to S✕T
    return R


class Render:
    def __init__(self, R: np.ndarray, params: Params) -> None:
        self.R = R
        self.params = params
        date = datetime.now().strftime("%Y%m%d")
        time = datetime.now().strftime("%H%M")
        self.dir_path = "output/{}/{}".format(date, time)
        os.makedirs(self.dir_path)

    def overview(self):
        fig = plt.figure(figsize=(20, 7))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Time [t]')
        ax.set_ylabel('Discretized Biomass')
        ax.set_xticks([i for i in range(self.params.n_timestep+1)])
        ax.set_yticks([i for i in range(self.params.n_digitize)])
        ax.imshow(self.R, cmap=self.params.color_thme)
        ax.invert_yaxis()

        axpos = ax.get_position()
        self.__set_colorbar(fig, ax, axpos)
        self.__set_title()
        file_path = '{}/overview.png'.format(self.dir_path)
        fig.savefig(file_path)

    def per_time(self):
        fig = plt.figure(figsize=(20, 7))
        for t in range(self.params.n_timestep):
            r = self.R[:, t:t+1]
            ax = plt.subplot(1, self.params.n_timestep, t+1)
            ax.axis('off')
            ax.imshow(r, cmap=self.params.color_thme)
            ax.invert_yaxis()
            plt.title(t)
            if t == self.params.n_timestep-1:
                axpos = ax.get_position()
        self.__set_colorbar(fig, ax, axpos)
        self.__set_title()
        file_path = '{}/per_time.png'.format(self.dir_path)
        fig.savefig(file_path)

    def differential_overview(self):
        fig = plt.figure(figsize=(20, 7))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('x (Biomass)')
        ax.set_ylabel('dx/dt')
        # delta = {s: [] for s in range(self.params.n_digitize)}

        S = np.array([s for s in range(self.params.n_digitize)])
        x = []
        y = []
        for t in range(self.params.n_timestep-1):
            r_t = self.R[:, t:t+1]
            r_next_t = self.R[:, t+1:t+2]
            r_t_probs = r_t/sum(r_t)
            r_next_t_probs = r_next_t/sum(r_next_t)
            s_t = np.argmax(r_t)
            # s_next_t = np.argmax(r_next_t)
            # r(s)を確率分布として考え、確率✕x(s)をs_tとすることで離散化の影響を小さくする
            # s_t_detailed = float(np.dot(S, r_t))
            s_t_detailed = float(np.dot(S, r_t_probs))
            # s_next_t_detailed = float(np.dot(S, r_next_t))
            s_next_t_detailed = float(np.dot(S, r_next_t_probs))
            s_delta = s_next_t_detailed-s_t_detailed
            # delta[s_t].append(s_delta)
            y.append(s_delta)
            x.append(s_t_detailed)
        # x = []
        # y = []
        # for k, v in delta.items():
        #     if v == []:
        #         continue
        #     x_i = int(k)
        #     y_i = sum(v)/len(v)
        #     x.append(x_i)
        #     y.append(y_i)
        ax.scatter(x, y, label="Relationships between x and dx/dt")
        plt.legend()
        self.__set_title()
        file_path = '{}/differential_overview.png'.format(self.dir_path)
        fig.savefig(file_path)
        self.actual_size_differential_overview(x, y)

    def actual_size_differential_overview(self, x, y):
        fig = plt.figure(figsize=(20, 7))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('x (Biomass)')
        ax.set_ylabel('dx/dt')
        ax.grid()
        dp = DataProcessor()
        data = dp.read_data(self.params.data_path)
        max_element = float(np.amax(data))
        size_ratio = max_element/self.params.n_digitize

        x = [x_i*size_ratio for x_i in x]
        y = [y_i*size_ratio for y_i in y]
        ax.scatter(x, y,
                   label="Relationships between x and dx/dt")
        # ax.plot(x, y,
        #         label="Relationships between x and dx/dt")
        x_answer = np.linspace(0, 1000, 1001)
        y_answer = 0.2*(1-x_answer/1000)*x_answer
        ax.plot(x_answer, y_answer, label="Answer", color="red")
        plt.legend()
        file_path = '{}/differential_actual_size_overview.png'.format(
            self.dir_path)
        fig.savefig(file_path)

    def __set_title(self):
        algorithm_name = "Maximum entropy deep IRL" if self.params.is_deep_learning else "Maximum entropy IRL"
        title = f"{algorithm_name} \n n_digitiz={self.params.n_digitize} n_timestep={self.params.n_timestep}, alpha={self.params.alpha}, n_epoch={self.params.n_epoch}"
        plt.suptitle(title)

    def __set_colorbar(self, fig, ax, axpos):
        axpos = ax.get_position()
        cbar_ax = fig.add_axes([0.87, axpos.y0, 0.02, axpos.height])
        norm = colors.Normalize(vmin=self.R.min(), vmax=self.R.max())
        mappable = ScalarMappable(cmap=self.params.color_thme, norm=norm)
        mappable._A = []
        fig.colorbar(mappable, cax=cbar_ax)
        plt.subplots_adjust(right=0.85)
        plt.subplots_adjust(wspace=0.1)
