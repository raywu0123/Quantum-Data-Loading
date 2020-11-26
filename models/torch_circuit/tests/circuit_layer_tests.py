import pytest
import torch
import numpy as np

from ...torch_circuit import ParallelRYComplex, EntangleComplex, ParallelRY, Entangle
from models.qae import Decoder

n_qubit = 3


@pytest.mark.parametrize("layer", [ParallelRY(n_qubit), Entangle(n_qubit)])
def test_unitary_matrix(layer):
    N = 10000

    x = torch.empty(N, 2 ** n_qubit).normal_()

    y = layer(x)
    op = layer.op.data

    torch.allclose(op.T @ op, torch.eye(2 ** n_qubit))

    x_norm = x.norm(p=2, dim=-1)
    y_norm = y.norm(p=2, dim=-1)
    assert torch.allclose(x_norm, y_norm)


@pytest.mark.parametrize("layer", [
    ParallelRYComplex(n_qubit), 
    EntangleComplex(n_qubit),
    Decoder(n_qubit, 3),
])
def test_unitary_complex_preserve_norm(layer):
    N = 10000

    x_real = torch.empty(N, 2 ** n_qubit).normal_()
    x_imag = torch.empty(N, 2 ** n_qubit).normal_()

    y_real, y_imag = layer((x_real, x_imag))

    x_norm = x_real.norm(p=2, dim=-1) ** 2 + x_imag.norm(p=2, dim=-1) ** 2
    y_norm = y_real.norm(p=2, dim=-1) ** 2 + y_imag.norm(p=2, dim=-1) ** 2
    assert torch.allclose(x_norm, y_norm)


@pytest.mark.parametrize("layer", [
    ParallelRY(n_qubit), 
    Entangle(n_qubit),
])
def test_unitary_linear(layer):
    N = 10000

    x1 = torch.empty(N, 2 ** n_qubit).normal_()
    y1 = layer(x1)

    x2 = torch.empty(N, 2 ** n_qubit).normal_()
    y2 = layer(x2)

    x3 = x1 + x2
    y3 = layer(x3)

    print(y3[0], y1[0], y2[0], (y1 + y2)[0])
    assert torch.allclose(y3, y1 + y2, atol=1e-3)


@pytest.mark.parametrize("layer", [
    ParallelRYComplex(n_qubit), 
    EntangleComplex(n_qubit),
    Decoder(n_qubit, 3),
])
def test_unitary_complex_linear(layer):
    N = 10000

    x1_real = torch.zeros(N, 2 ** n_qubit)
    x1_real[:, 0] = 1.
    x1_imag = torch.empty(N, 2 ** n_qubit).normal_()
    y1_real, y1_imag = layer((x1_real, x1_imag))

    x2_real = torch.zeros(N, 2 ** n_qubit)
    x2_real[:, 1] = 1.
    x2_imag = torch.empty(N, 2 ** n_qubit).normal_()
    y2_real, y2_imag = layer((x2_real, x2_imag))

    x3_real, x3_imag = x1_real + x2_real, x1_imag + x2_imag
    y3_real, y3_imag = layer((x3_real, x3_imag))

    assert torch.allclose(y3_real, y1_real + y2_real, atol=1e-3)
    assert torch.allclose(y3_imag, y1_imag + y2_imag, atol=1e-3)
