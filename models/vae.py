import torch
from torch import nn
import numpy as np

from .base import ModelBaseClass
from .utils import DataGenerator, sample_from
from utils import epsilon, ints_to_bits


class Encoder(nn.Module):

    def __init__(self, z_dim: int, data_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Embedding(num_embeddings=2 ** data_dim, embedding_dim=z_dim),
            nn.Linear(z_dim, z_dim),
            nn.LeakyReLU(),
        )
        self.mu_linear = nn.Linear(z_dim, z_dim)
        self.logvar_linear = nn.Linear(z_dim, z_dim)

    def forward(self, x):
        x = self.layers(x)
        return self.mu_linear(x), self.logvar_linear(x)


class Decoder(nn.Module):

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
    


class VAE(ModelBaseClass):

    def __init__(self, n_qubit: int, batch_size: int, n_epoch: int, **kwargs):
        self.n_qubit = n_qubit
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.device = 'gpu:0' if torch.cuda.is_available() else 'cpu'
        self.z_dim = n_qubit
        self.encoder = Encoder(self.z_dim, n_qubit).to(self.device)
        self.decoder = Decoder(self.z_dim, n_qubit).to(self.device)
        self.opt = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=1e-2
        )

    def fit(self, data: np.array) -> np.array:
        for i_epoch in range(self.n_epoch):
            kl_losses, recon_losses = [], []
            verbose = True
            for real_batch in DataGenerator(data, self.batch_size):
                kl_loss, recon_loss = self.fit_batch(real_batch, verbose=verbose)
                verbose = False
                kl_losses.append(kl_loss)
                recon_losses.append(recon_loss)

            print(f'{i_epoch} {np.mean(kl_losses):4f} {np.mean(recon_losses):4f}')
        
        with torch.no_grad():
            random_z = torch.randn([len(data), self.z_dim], device=self.device).float()
            probs = self.decoder(random_z).mean(dim=0)

        return probs.cpu().data.numpy()
        

    def fit_batch(self, batch: np.array, verbose: bool):
        batch = torch.from_numpy(batch).to(self.device).long() 
        mu, logvar = self.encoder(batch)
        std = torch.exp(0.5 * logvar)
        e = torch.randn_like(mu)
        z = mu + e * std
        recon = self.decoder(z)

        recon_loss_fn = nn.NLLLoss(reduction='none')
        recon_loss = recon_loss_fn(torch.log(recon + epsilon), batch)
        
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss.mean() + kl_loss
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return kl_loss.item(), recon_loss.mean().item()
