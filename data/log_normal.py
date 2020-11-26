from .base import DataBaseClass

import numpy as np


class LogNormal(DataBaseClass):

    mu, sigma = 1., 1.

    def __init__(self, range_: int):
        super().__init__(range_)

    def get_data(self, num: int) -> np.array:
        ds = []
        while len(ds) < num:
            ds.extend([
                d
                for d in np.random.lognormal(self.mu, self.sigma, num)
                if 0 <= d <= self.range - 1
            ])
        return np.round(ds[:num]).astype(int)
