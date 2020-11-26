from abc import ABC, abstractmethod

import numpy as np


class DataBaseClass(ABC):

    def __init__(self, range_: int):
        self.range = range_

    @abstractmethod
    def get_data(self, num: int) -> np.array:
        pass
