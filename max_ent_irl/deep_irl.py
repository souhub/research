import chainer
from irl import MaxEntIRL

from chainer import Chain
from chainer.optimizer import WeightDecay, GradientClipping
import numpy as np
import chainer.links as L
import chainer.functions as F
from chainer import optimizers
import gym
from chainer.optimizers import ada_grad


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
    def __init__(self, env: gym.Env, alpha: float, epoch: int) -> None:
        super().__init__(env, alpha, epoch)

    def irl(self, trajectories: np.ndarray):
        features = np.eye(self.env.observation_space.n)
        expert_svf = self.compute_expert_features(trajectories)
        x = chainer.Variable(features.astype(np.float32))

        reward_func = Reward(self.env, 64)
        optimizer = optimizers.Adam(alpha=self.alpha)
        # optimizer = optimizers.AdaGrad(lr=self.alpha)
        optimizer.setup(reward_func)
        optimizer.add_hook(WeightDecay(1e-4))
        optimizer.add_hook(GradientClipping(100.0))

        print("Learning in progress...")
        for e in range(self.epoch):
            reward_func.zerograds()
            r = reward_func(x)
            self.R = r
            V, policy = self.planner.plan()
            # Compute a SVF using the policy.
            svf = self.compute_svf(trajectories, policy)
            grad_r = expert_svf-svf
            r.grad = - \
                grad_r.reshape((self.env.observation_space.n, 1)
                               ).astype(np.float32)
            r.backward()
            optimizer.update()
            if e != 0 and e % 100 == 0:
                print("     {}/{} epoches have been completed...".format(e, self.epoch))
        return reward_func(x).data.reshape((self.env.observation_space.n,))
