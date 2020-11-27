from functools import partial

import numpy as np
import torch
from torch import nn

from utils import bits_to_ints, epsilon, evaluate, get_pmf
from .base import ModelBaseClass
from .utils import EMA, DataGenerator, sample_from, counts
from .torch_circuit import ParallelRY, Entangle


class Generator(nn.Module):

    def __init__(self, n_qubit: int, k: int):
        super().__init__()
        self.n_qubit = n_qubit
        self.preparation_layer = ParallelRY(n_qubit)

        self.layers = nn.ModuleList()
        for _ in range(k):
            self.layers.append(Entangle(n_qubit))
            self.layers.append(ParallelRY(n_qubit))

    def forward(self, x):
        x = self.preparation_layer(x)
        for layer in self.layers:
            x = layer(x)
        probs = torch.abs(x) ** 2
        return probs


class MMD(nn.Module):

    def __init__(self, sigmas: list, n_qubit: int):
        super().__init__()
        self.n_qubit = n_qubit
        self.K = nn.Parameter(self.make_K(sigmas), requires_grad=False)

    def forward(self, x, y):
        x_y = (x - y).unsqueeze(-1)
        return x_y.T @ self.K @ x_y

    def to_binary(self, x):
        r = torch.arange(self.n_qubit)
        to_binary_op = torch.ones_like(r) << r  # (n_qubit,)
        return ((x.unsqueeze(-1) & to_binary_op) > 0).long()

    def make_K(self, sigmas: list):
        sigmas = torch.Tensor(sigmas)
        r = self.to_binary(torch.arange(2 ** self.n_qubit)).float()  # (2 ** n_qubit, n_qubit)

        x = r.unsqueeze(1)  # (2 ** n_qubit, 1, n_qubit)
        y = r.unsqueeze(0)  # (1, 2 ** n_qubit, n_qubit)
        
        x_y = torch.einsum('abn,bcn->acn', x, y)  
        norm_square = (x** 2 + y ** 2 - 2 * x * y).sum(dim=-1)  # (2 ** n_qubit, 2 ** n_qubit)
        
        K = (-norm_square.unsqueeze(-1) / (2 * sigmas)).exp().sum(dim=-1)  # (2 ** n_qubit, 2 ** n_qubit)
        return K


class QCBM(ModelBaseClass):

    def __init__(self, n_qubit: int, batch_size: int, n_epoch: int, **kwargs):
        self.n_qubit = n_qubit
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        self.ema = EMA(0.9).to(self.device)
        self.mmd = MMD([0.5, 1., 2., 4.], n_qubit=n_qubit).to(self.device)
        
        self.generator = Generator(self.n_qubit, k=3).to(self.device)
        self.optim = torch.optim.Adam(params=self.generator.parameters(), lr=1e-2)

    def fit(self, data: np.array) -> np.array:
        data_counts = counts(data, self.n_qubit)
        
        data_pmf = get_pmf(data_counts)
        data_pmf = torch.from_numpy(data_pmf).float().to(self.device)
        for i_epoch in range(self.n_epoch):
            mmd_losses = []
            for _ in DataGenerator(data, self.batch_size):
                mmd_loss = self.train_generator(data_pmf)
                mmd_losses.append(mmd_loss)
            
            eval_results = evaluate(data_counts, self.get_outcome())
            print(f'{i_epoch} MMD: {np.mean(mmd_losses):4f}　KL: {eval_results["kl"]:4f} JS: {eval_results["js"]:4f}')
        
        return self.get_outcome()

    def get_outcome(self):
        z = self.get_prior()
        with torch.no_grad():
            gen_probs = self.generator(z)
        return gen_probs[0].cpu().data.numpy()

    def train_generator(self, data_pmf: torch.Tensor):
        z = self.get_prior().repeat(self.batch_size, 1)
        fake_probs = self.generator(z)

        fake_data = sample_from(fake_probs)      
        selected_probs = torch.gather(fake_probs, dim=-1, index=fake_data)
        log_selected_probs = torch.log(selected_probs + epsilon).squeeze(dim=-1)

        fake_data_pmf = (torch.arange(2 ** self.n_qubit) == fake_data).sum(dim=0).float() / self.batch_size
        
        mmd = self.mmd(data_pmf, fake_data_pmf)
        reward = -mmd
        baseline = self.ema(reward.mean().data)

        loss = (-(reward - baseline) * log_selected_probs).mean()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return mmd.item()

    def get_prior(self) -> torch.Tensor:
        z = torch.zeros([1, 2 ** self.n_qubit]).to(self.device)
        z[:, 0] = 1
        # prepare initial state at "0"
        return z