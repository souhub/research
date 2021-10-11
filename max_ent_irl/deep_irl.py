import chainer
from irl import MaxEntIRL

import chainer
from chainer import Chain
from chainer.optimizer import WeightDecay, GradientClipping
import numpy as np
import chainer.links as L
import chainer.functions as F
from chainer import optimizers
import gym


class Reward(Chain):
    def __init__(self, env: gym.Env, n_hidden: int):
        super(Reward, self).__init__(
            # 入力されるノード数がn_features個、出力されるノード数がn_hidden個
            l1=L.Linear(env.observation_space.n, n_hidden),
            l2=L.Linear(n_hidden, n_hidden),
            l3=L.Linear(n_hidden, 1)
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


class MaxEntDeepIRL(MaxEntIRL):
    def __init__(self, env: gym.Env, alpha: float) -> None:
        super().__init__(env, alpha)

    def irl(self, trajectories: np.ndarray, epoch: int = 100):
        features = np.eye(self.env.observation_space.n)
        expert_features = self.compute_expert_features(trajectories)
        x = chainer.Variable(features.astype(np.float32))
        reward_func = Reward(self.env, 64)
        optimizer = optimizers.AdaGrad(lr=self.alpha)
        r = reward_func(x)
        print(type(r))
        optimizer.setup(reward_func)
        optimizer.add_hook(WeightDecay(1e-4))
        optimizer.add_hook(GradientClipping(100.0))

        for _ in range(epoch):
            reward_func.zerograds()
            r = reward_func(x)
            self.R = r
            V, policy = self.planner.plan()
            # Compute a SVF using the policy.
            svf = self.compute_svf(trajectories, policy)
            grad_r = expert_features-svf
            r.grad = - \
                grad_r.reshape((self.env.observation_space.n, 1)
                               ).astype(np.float32)
            r.backward()
            optimizer.update()
        return reward_func(x).data.reshape((self.env.observation_space.n))


if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    n_hidden = 3
    r = Reward(env, n_hidden)
    print(r())
