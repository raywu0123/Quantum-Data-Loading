from .base import DataBaseClass

import numpy as np


class Triangular(DataBaseClass):

    left, mode, right = 0, 2, 7

    def __init__(self, range_: int):
        super().__init__(range_)

    def get_data(self, num: int) -> np.array:
        ds = []
        while len(ds) < num:
            ds.extend([
                d
                for d in np.random.triangular(self.left, self.mode, self.right, num)
                if 0 <= d <= self.range - 1
            ])
        return np.round(ds[:num]).astype(int)
