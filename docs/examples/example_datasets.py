import numpy as np
import pandas as pd
import scipy
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from plotting import remove_spines

SEED = 42

class _Dataset(ABC):

    def __init__(self, noise_sigma=0.):
        self.noise_sigma = noise_sigma

    def plot(self, ax=None, nsamp=5, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(4, 3))

        in_params = np.linspace(*self.in_params_range, 1000)
        out_params = self.true(in_params)
        samples = self.sample(nsamp, seed=SEED)

        if 'color' not in kwargs:
            kwargs['color'] = 'black'

        ax.plot(in_params, out_params, **kwargs, label='True')
        ax.scatter(samples[0], samples[1], **kwargs, label='Training Data')
        remove_spines(ax)

        # place legemnd on right of plot
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)

        return ax

    @abstractmethod
    def true(self, in_params) -> np.array:
        pass

    @property
    @abstractmethod
    def in_params_range(self):
        pass

    @property
    def out_params_range(self):
        return self.true(self.in_params_range[0]), self.true(self.in_params_range[1])

    def sample(self, n, seed=None):
        if seed is not None:
            np.random.seed(seed)
        in_params = np.random.uniform(*self.in_params_range, n)
        return in_params, self(in_params)

    def __call__(self, in_params):
        if isinstance(in_params, (int, float)):
            in_params = np.array([in_params])
        n = len(in_params)
        noise = np.random.normal(0, self.noise_sigma, n)
        return self.true(in_params) + noise


class Dataset1(_Dataset):
    def true(self, in_params):
        x = in_params
        return 1 / (x ** 2 + 1) * np.cos(np.pi * x)

    @property
    def in_params_range(self):
        return (-2, 5)




class Dataset2(_Dataset):
    def true(self, in_params) -> np.array:
        x = in_params
        return 0.2 + 0.4 * x ** 2 + 0.3 * x * np.sin(15 * x) + 0.05 * np.cos(50 * x)

    @property
    def in_params_range(self):
        return (0, 1)


class Dataset3(_Dataset):
    def true(self, in_params) -> np.array:
        x = in_params
        return np.sin(x) / 2 - ((10 - x) ** 2) / 50 + 2

    @property
    def in_params_range(self):
        return (0, 20)

