import numpy as np
import torch
from torch import nn

from utils import epsilon


def DataGenerator(data: np.array, batch_size: int):
    np.random.shuffle(data)
    num_batch = len(data) // batch_size
    for i in range(num_batch):
        yield data[i * batch_size: (i + 1) * batch_size]


def counts(ints: np.array, n_bit: int):
    ret = np.zeros([2 ** n_bit])
    for i in ints:
        ret[i] += 1
    return ret


def sample_from(probs: torch.Tensor) -> torch.Tensor:
    m = torch.distributions.Categorical(probs=probs)
    sample = m.sample().view(-1, 1)
    return sample


class EMA(nn.Module):
    
    def __init__(self, mu):
        super(EMA, self).__init__()
        self.mu = mu
        self.last_average = 0.
        
    def forward(self, x):
        self.last_average = self.mu * x + (1 - self.mu) * self.last_average
        return self.last_average
