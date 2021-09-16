import gym
import numpy as np


# 各値を離散値に変換
def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num+1)[1:-1]


def digitize_state(observation):
    num_dizitized = 4
    position, velocity = observation
    digitized = [
        np.digitize(position, bins=bins(-1.2, 0.6, num_dizitized)),
        np.digitize(velocity, bins=bins(-0.07, 0.07, num_dizitized)),
    ]
    return sum([x * (num_dizitized**i) for i, x in enumerate(digitized)])


if __name__ == '__main__':
    d = bins(-1.2, 6, 10)
    print(d, len(d))
