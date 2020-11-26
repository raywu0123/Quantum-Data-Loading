from functools import reduce

import torch
from torch import nn, autograd
import numpy as np

from .base import ModelBaseClass
from .utils import DataGenerator, sample_from
from .torch_circuit import (
    ParallelRYComplex, 
    EntangleComplex, 
    Exp, 
    batch_kronecker_complex,
)
from utils import epsilon, ints_to_onehot


class Encoder(nn.Module):

    def __init__(self, z_dim: int, data_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Embedding(num_embeddings=2 ** data_dim, embedding_dim=z_dim),
            nn.Linear(z_dim, z_dim),
            nn.LeakyReLU(),
        )
        self.theta_linear = nn.Linear(z_dim, z_dim)
        self.phi_linear = nn.Linear(z_dim, z_dim)
        self.log_kappa_linear = nn.Linear(z_dim, z_dim)

    def forward(self, x):
        x = self.layers(x)
        return (
            torch.sigmoid(self.theta_linear(x)) * np.pi, 
            torch.sigmoid(self.phi_linear(x)) * 2 * np.pi,
            torch.exp(self.log_kappa_linear(x)).clamp(max=100.),
        )



class Decoder(nn.Module):

    def __init__(self, n_qubit: int, k: int):
        super().__init__()
        self.preparation_layer = ParallelRYComplex(n_qubit)
        self.layers = nn.ModuleList()
        for _ in range(k):
            self.layers.append(EntangleComplex(n_qubit))
            self.layers.append(ParallelRYComplex(n_qubit))

    def forward(self, x):
        x = self.preparation_layer(x)
        for layer in self.layers:
            x = layer(x)

        probs = x[0] ** 2 + x[1] ** 2
        return probs


def householder_transform(z_: torch.Tensor, mu: torch.Tensor):
    N, n_qubit = z_.shape[:2]
    z_ = z_.view(N * n_qubit,  3)
    mu = mu.view(N * n_qubit, 3)
    
    e1 = torch.zeros_like(mu)
    e1[:, 0] = 1

    u_ = e1 - mu
    u = u_ / (u_.norm(p=2, dim=1, keepdim=True) + epsilon)  # (N, n_qubit * 3)
    uuT = torch.einsum('na,nb->nab', u, u)
    uuT_z_ = torch.einsum('nba,na->nb', uuT, z_)
    
    return (z_ - 2 * uuT_z_).view(N, n_qubit, 3)


class Where(autograd.Function):

    @staticmethod
    def forward(ctx, cond, x, f1, f2):
        with torch.enable_grad():
            y1 = f1(x)
            y1_sum = y1.sum()
            y2 = f2(x)
            y2_sum = y2.sum()
            z = torch.where(cond, y1, y2)
            
        ctx.save_for_backward(cond, x, y1_sum, y2_sum)
        return z

    @staticmethod
    def backward(ctx, grad_input):
        cond, x, y1_sum, y2_sum = ctx.saved_tensors
        with autograd.set_detect_anomaly(False):
            g1 = autograd.grad(y1_sum, x)[0]
            g2 = autograd.grad(y2_sum, x)[0]

        g1[torch.isnan(g1)] = 0.
        g1[torch.isinf(g1)] = 0.
        g2[torch.isnan(g2)] = 0.
        g2[torch.isinf(g2)] = 0.
        
        g = torch.where(cond, g1, g2) * grad_input
        return None, g, None, None
        
where = Where.apply


def calculate_kl_loss(kappa: torch.Tensor):
    # kappa in [0, inf)
    kl_small = lambda k : (k + epsilon) / torch.tanh(k + epsilon) + torch.log((k + epsilon) / torch.sinh(k + epsilon)) - 1
    kl_large = lambda k : torch.log(k) + np.log(2) - 1
    kl = where(kappa < 50., kappa, kl_small, kl_large)
    return kl.sum(dim=-1)


def sample_omega(kappa: torch.Tensor):
    u = torch.rand_like(kappa)
    omega_tiny = lambda k : 2 * u - 1 - 2 * u * (u - 1) * k
    omega_normal = lambda k : torch.log((torch.exp(2 * k) - 1) * u + 1) / (k + epsilon) - 1    
    omega_large = lambda k : 1 + torch.log(u + epsilon) / k
    omega = where(
        kappa < 1e-5, kappa, 
        omega_tiny,
        lambda k : where(kappa < 40., k, omega_normal, omega_large),
    )
    return omega


class QVAE(ModelBaseClass):

    def __init__(self, n_qubit: int, batch_size: int, n_epoch: int, **kwargs):
        self.n_qubit = n_qubit
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.device = 'gpu:0' if torch.cuda.is_available() else 'cpu'
        self.encoder = Encoder(n_qubit, 3).to(self.device)
        self.decoder = Decoder(n_qubit, 3).to(self.device)
        self.opt = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=1e-2
        )

    def fit(self, data: np.array) -> np.array:
        for i_epoch in range(self.n_epoch):
            kl_losses, recon_losses = [], []
            verbose = True
            for real_batch in DataGenerator(data, self.batch_size):
                with autograd.detect_anomaly():
                    kl_loss, recon_loss = self.fit_batch(real_batch, verbose=verbose)
                verbose = False
                kl_losses.append(kl_loss)
                recon_losses.append(recon_loss)

            print(f'{i_epoch:3d} KL: {np.mean(kl_losses):4f} RECON: {np.mean(recon_losses):4f}')
        
        with torch.no_grad():
            # kronecker product of |+> states
            z_real = torch.ones([1, 2 ** self.n_qubit]) / (np.sqrt(2) ** self.n_qubit)
            z_imag = torch.zeros([1, 2 ** self.n_qubit])
            probs = self.decoder((z_real, z_imag))[0]

        return probs.cpu().data.numpy()
        

    def fit_batch(self, batch: np.array, verbose: bool):
        batch = torch.from_numpy(batch).to(self.device).long()
        mu_theta, mu_phi, kappa = self.encoder(batch)
        
        mu = torch.cat([
            (torch.sin(mu_theta) * torch.cos(mu_phi)).unsqueeze(-1),
            (torch.sin(mu_theta) * torch.sin(mu_phi)).unsqueeze(-1),
            torch.cos(mu_phi).unsqueeze(-1),
        ], dim=-1)
        # (N, n_qubit, 3)
        omega = sample_omega(kappa).unsqueeze(-1)
        v = torch.rand_like(omega) * 2 * np.pi
        # (N, n_qubit, 1)
        z_ = torch.cat([
            omega,
            torch.sqrt(1 - omega ** 2 + epsilon) * torch.cos(v),
            torch.sqrt(1 - omega ** 2 + epsilon) * torch.sin(v),
        ], dim=-1)
        # (N, n_qubit, 3)
        z = householder_transform(z_, mu)  # (N, n_qubit, 3)
        zx, zy, zz = z[..., 0], z[..., 1], z[..., 2]   # (N, n_qubit)

        cos_theta_half = ((1 + zz) / 2 + epsilon).sqrt()
        sin_theta_half = ((1 - zz) / 2 + epsilon).sqrt()
        cos_phi = zx / ((zx ** 2 + zy ** 2) + epsilon)
        cos_phi = cos_phi.clamp(-1, 1) 
        sin_phi = 1 - cos_phi ** 2
        # (N, n_qubit)

        x_real = cos_theta_half
        x_imag = torch.zeros_like(x_real)
        y_real = sin_theta_half * cos_phi
        y_imag = sin_theta_half * sin_phi

        single_qubit_states_real = torch.cat([
            x_real.T.unsqueeze(-1),
            y_real.T.unsqueeze(-1),
        ], dim=-1)  # (n_qubit, N, 2)
        single_qubit_states_imag = torch.cat([
            x_imag.T.unsqueeze(-1),
            y_imag.T.unsqueeze(-1),
        ], dim=-1)
        
        z = reduce(
            lambda x, y: batch_kronecker_complex(x, y), 
            zip(single_qubit_states_real, single_qubit_states_imag)
        )  # Tuple[(N, 2 ** n_qubit)]
        recon = self.decoder(z)
        recon_loss_fn = nn.NLLLoss(reduction='none')
        recon_loss = recon_loss_fn(torch.log(recon + epsilon), batch)
        
        kl_loss = calculate_kl_loss(kappa)
        loss = recon_loss.mean() + kl_loss.clamp(min=10.).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return kl_loss.mean().item(), recon_loss.mean().item()
