import numpy as np
import torch
import pytest

from ..qvae import calculate_kl_loss, householder_transform, sample_omega, where


@pytest.mark.parametrize(
    "x,expected_kl", [
       (0., 0.),
       (49.999, 3.6051),
       (50.001, 3.6051),
    ])
def test_calculate_kl_loss(x: float, expected_kl: float):
    x = torch.ones([1]) * x
    kl = calculate_kl_loss(x)
    assert kl.item() == pytest.approx(expected_kl, rel=1e-4)
    

def test_householder_transform():
    N, n_qubit = 1, 3

    # z_ = e1
    z_ = torch.zeros([N, n_qubit, 3])
    z_[..., 0] = 1

    mu = torch.Tensor([
        [1, 2, 3],
        [2, 2, 3],
        [3, 3, 3]
    ])
    mu = (mu / mu.norm(p=2, dim=-1, keepdim=True)).view([N, n_qubit, 3])
    
    z = householder_transform(z_, mu)

    z = z.data.numpy()
    mu = mu.data.numpy()
    np.testing.assert_almost_equal(z, mu)


def test_custom_where_autograd_function():
    x = torch.Tensor([0., 1., 2.])
    x.requires_grad = True

    f1 = lambda x_ : 2 * x_
    f2 = lambda x_ : 1 / x_
    z = where(x == 0., x, f1, f2)
    z.sum().backward()

    grad = x.grad.data.numpy()
    expected_grad = np.array([2., -1., -0.25])
    np.testing.assert_almost_equal(grad, expected_grad)


@pytest.mark.parametrize("kappa", [0., 1., 10., 30., 50., 100.])
def test_sample_omega(kappa: float):
    N = 10000
    kappa = torch.ones(N, requires_grad=True) * kappa
    kappa.retain_grad()
    omega = sample_omega(kappa)

    assert omega.max().item() == pytest.approx(1., rel=1e-2)
    assert omega.min().item() >= -1.
    assert not torch.isnan(omega).any().item()
    assert not torch.isinf(omega).any().item()
    
    with torch.autograd.detect_anomaly():
        omega.mean().backward()
        
    assert not torch.isnan(kappa.grad).any().item()