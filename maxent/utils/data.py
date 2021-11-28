import numpy as np
import pandas as pd
from typing import List


class DataProcessor:
    def read_data(self, data_path: str) -> np.ndarray:
        df = pd.read_csv(data_path, header=None,
                         index_col=0, skiprows=7)
        return np.array(df)

    def digitize(self, data: np.ndarray, n_state: int, limit: tuple) -> np.ndarray:
        low, high = limit
        bins = np.linspace(low, high, n_state)
        return np.digitize(data, bins=bins)

    def save_data(self, data: np.ndarray, save_path: str) -> None:
        np.savetxt(save_path, data, fmt="%08f")

    def save_numpy_data(self, data: np.ndarray, save_path: str) -> None:
        np.save(save_path, data)

    def read_numpy_data(self, data_path: str) -> np.ndarray:
        return np.load(data_path)

    def read_txt(self, data_path: str) -> List[float]:
        return np.loadtxt(data_path)


def get_trajectories(input_data: str, n_states: List[int], state_limits) -> List[np.array]:
    dp = DataProcessor()
    _data = dp.read_data(input_data)
    data = _data[:, :52].T
    _tempreture = data[0]
    _biomasses = data[1:, :]
    tempreture = dp.digitize(_tempreture, n_states[1], state_limits[1])
    trajectories = []
    for i in range(len(_biomasses)):
        _biomass = _biomasses[i]
        biomass = dp.digitize(_biomass, n_states[0], state_limits[0])
        traj = [biomass, tempreture]
        trajectories.append(traj)
    return trajectories
