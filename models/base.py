from abc import ABC, abstractmethod

import numpy as np


class ModelBaseClass(ABC):

    @abstractmethod
    def fit(self, data: np.array):
        pass
