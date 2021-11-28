import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.colors as colors


class Render:
    def __init__(self) -> None:
        self.cmap = 'viridis'

    def show(self):
        plt.show()

    def imshow(self, data: np.ndarray):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.imshow(data)
        ax.invert_yaxis()
        self.__set_colorbar(fig, ax, data)
        self.__set_label(ax)
        self.__set_title('Reward function R(T,B)')

    def scatter3d(self, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x, y, z, s=10)
        self.__set_label(ax)
        self.__set_title('Biomass variation(B,T)')

    def __set_title(self, title: str):
        plt.suptitle(title)

    def __set_label(self, ax, x_label='Biomass (B)', y_label='Tempreture (T)'):
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    def __set_colorbar(self, fig, ax, vals: np.ndarray):
        axpos = ax.get_position()
        cbar_ax = fig.add_axes([0.87, axpos.y0, 0.02, axpos.height])
        norm = colors.Normalize(
            vmin=vals.min(), vmax=vals.max())
        mappable = ScalarMappable(norm=norm, cmap=self.cmap)
        mappable._A = []
        fig.colorbar(mappable, cax=cbar_ax)
        plt.subplots_adjust(right=0.85)
        plt.subplots_adjust(wspace=0.1)
