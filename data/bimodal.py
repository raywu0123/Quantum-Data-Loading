import random

import numpy as np

from .base import DataBaseClass


class Bimodal(DataBaseClass):

    mu1, sigma1 = .5, 1.
    mu2, sigma2 = 3.5, .5

    def __init__(self, range_: int):
        super().__init__(range_)

    def get_point(self):
        return [
            np.random.normal(self.mu1, self.sigma1, 1)[0],
            np.random.normal(self.mu2, self.sigma2, 1)[0],
        ][random.randint(0, 1)]

    def get_data(self, num: int) -> np.array:
        ds = []
        while len(ds) < num:
            p = self.get_point()
            if 0 <= p < self.range - 1:
                ds.append(p)
        return np.round(ds).astype(np.int)
