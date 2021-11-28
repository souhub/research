import numpy as np
import pandas as pd


class DataProcessor:
    def digitize(self, data: np.ndarray, n_digitize: int) -> np.ndarray:
        max_element = np.amax(data)
        min_element = 0
        bins = np.linspace(min_element, max_element, n_digitize-1)
        digitized_data = np.digitize(data, bins=bins)
        return digitized_data

    def read_data(self, data_path: str) -> np.ndarray:
        df = pd.read_csv(data_path, header=None, index_col=0)
        return np.array(df)

    def save_data_to_csv(self, data: np.ndarray, save_path: str) -> None:
        np.savetxt(save_path, data, fmt="%08f")
