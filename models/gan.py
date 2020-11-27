import numpy as np
import torch
from torch import nn

from utils import bits_to_ints, epsilon
from .base import ModelBaseClass
from .utils import DataGenerator, counts, sample_from, EMA


class Generator(nn.Module):

    def __init__(self, z_dim: int, data_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(z_dim, 2 ** data_dim),
            nn.LeakyReLU(),
            nn.Linear(2 ** data_dim, 2 ** data_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, z: torch.Tensor):
        return self.layers(z)


class Discriminator(nn.Module):

    def __init__(self, data_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Embedding(num_embeddings=2 ** data_dim, embedding_dim=data_dim),
            nn.Linear(data_dim, data_dim),
            nn.LeakyReLU(),
            nn.Linear(data_dim, 1),
        )

    def forward(self, batch: torch.Tensor):
        return self.layers(batch)


class GAN(ModelBaseClass):

    def __init__(self, n_qubit: int, batch_size: int, n_epoch: int, **kwargs):
        self.n_qubit = n_qubit
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.generator = Generator(
            z_dim=self.n_qubit,
            data_dim=self.n_qubit,
        ).to(self.device)
        self.ema = EMA(0.9).to(self.device)
        self.discriminator = Discriminator(self.n_qubit).to(self.device)
        self.g_optim = torch.optim.Adam(params=self.generator.parameters(), lr=1e-2)
        self.d_optim = torch.optim.Adam(params=self.discriminator.parameters(), lr=1e-2)
        self.z = self.generate_prior(self.n_qubit).to(self.device)

    def fit(self, data: np.array) -> np.array:
        for i_epoch in range(self.n_epoch):
            g_losses, d_losses = [], []
            for real_batch in DataGenerator(data, self.batch_size):
                g_loss = self.train_generator()
                g_losses.append(g_loss)
                d_loss = self.train_discriminator(real_batch)
                d_losses.append(d_loss)

            print(f'{i_epoch} {np.mean(g_losses):4f} {np.mean(d_losses):4f}', end=' ')

            # Ref: https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf
            mean_d_loss = np.mean(d_losses)
            print(f'JS: {np.log(2) - mean_d_loss / 2:.4f}')

        z = self.get_prior(1)
        with torch.no_grad():
            gen_probs = self.generator(z)

        return gen_probs[0].cpu().data.numpy()

    def train_generator(self):
        z = self.get_prior(self.batch_size)
        fake_probs = self.generator(z)
        fake_data = sample_from(fake_probs)
        fake_score = self.discriminator(fake_data)
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
        reward = loss_fn(fake_score, torch.zeros_like(fake_score))

        selected_probs = torch.gather(fake_probs, dim=-1, index=fake_data)
        # no log as paper

        baseline = self.ema(reward.mean())
        advantage = reward - baseline
        g_loss = -(advantage.detach() * selected_probs).mean()   # policy gradient
        self.g_optim.zero_grad()
        g_loss.backward()
        self.g_optim.step()
        return g_loss.item()

    def train_discriminator(self, real_batch: np.array):
        z = self.get_prior(self.batch_size)
        fake_probs = self.generator(z)
        fake_data = sample_from(fake_probs)
        fake_score = self.discriminator(fake_data)

        real_batch = torch.from_numpy(real_batch).to(self.device).long()
        real_score = self.discriminator(real_batch)

        loss_fn = torch.nn.BCEWithLogitsLoss()
        d_loss = loss_fn(real_score, torch.ones_like(real_score)) + loss_fn(fake_score, torch.zeros_like(fake_score))
        self.d_optim.zero_grad()
        d_loss.backward()
        self.d_optim.step()
        return d_loss.item()
    
    @staticmethod
    def generate_prior(n_qubit: int) -> torch.Tensor:
        return torch.from_numpy(np.random.normal(0, 1, [n_qubit])).float()
        
    def get_prior(self, size: int) -> torch.Tensor:
        return self.z.repeat((size, 1))
