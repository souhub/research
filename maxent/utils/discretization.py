import numpy as np
from typing import List


def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num)[1:-1]


def discrete_state(_state, n_states: List[int], _state_limits=List[float]):
    state = []
    for n in range(len(n_states)):
        min, max = _state_limits[n]
        n_state = n_states[n]
        discreted_state = np.digitize(
            _state[n], bins=bins(min, max, n_state))
        state.append(discreted_state)
    return tuple(state)
