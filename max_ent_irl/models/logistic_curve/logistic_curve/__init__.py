from gym.envs.registration import register

register(
    id='logistic-curve',
    entry_point='gym_logistic_curve.envs.LogisticCurve'
)
