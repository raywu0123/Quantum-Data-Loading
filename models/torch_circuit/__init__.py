from functools import reduce

import numpy as np
import torch
from torch import nn


def kronecker(A, B):
    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0),  A.size(1)*B.size(1))


def batch_kronecker(A, B):
    return torch.einsum("na,nb->nab", A, B).view(A.size(0), A.size(1) * B.size(1))


def batch_kronecker_complex(A: tuple, B: tuple):
    A_r, A_i = A
    B_r, B_i = B
    return (
        batch_kronecker(A_r, B_r) - batch_kronecker(A_i, B_i),
        batch_kronecker(A_i, B_r) + batch_kronecker(A_r, B_i),
    )


class ParallelRY(nn.Module):

    def __init__(self, n_qubit: int):
        super().__init__()
        self.n_qubit = n_qubit
        self.params = nn.parameter.Parameter(
            (torch.rand([n_qubit]) * 2 - 1) * np.pi
        )

    def forward(self, x: torch.Tensor):
        cos = torch.cos(self.params).view(self.n_qubit, 1)
        sin = torch.sin(self.params).view(self.n_qubit, 1)
        single_qubit_gates = torch.cat([cos, -sin, sin, cos], dim=1).view(self.n_qubit, 2, 2)
        
        self.op = reduce(lambda x, y: kronecker(x, y), single_qubit_gates)
        return x @ self.op


class ParallelRYComplex(nn.Module):

    def __init__(self, n_qubit: int):
        super().__init__()
        self.n_qubit = n_qubit
        self.params = nn.Parameter(
            (torch.rand([n_qubit]) * 2 - 1) * np.pi
        )

    def forward(self, x: tuple):
        cos = torch.cos(self.params).view(self.n_qubit, 1)
        sin = torch.sin(self.params).view(self.n_qubit, 1)
        single_qubit_gates = torch.cat([cos, -sin, sin, cos], dim=-1).view(self.n_qubit, 2, 2)
        op = reduce(lambda a, b: kronecker(a, b), single_qubit_gates)
        return (x[0] @ op, x[1] @ op)


class Entangle(nn.Module):

    def __init__(self, n_qubit: int):
        super().__init__()
        gates = []
        dim = 2 ** n_qubit
        for i in range(n_qubit):
            c_idx, v_idx = i, (i + 1) % n_qubit
            g = torch.eye(dim)
            for j in range(dim):
                if (j // 2 ** c_idx) % 2 == 1 and (j // 2 ** v_idx) % 2 == 1:
                    g[j, j] = -1            
            gates.append(g)

        self.op = nn.Parameter(reduce(lambda x, y: x * y, gates), requires_grad=False)

    def forward(self, x: torch.Tensor):
        return x @ self.op


class EntangleComplex(nn.Module):

    def __init__(self, n_qubit: int):
        super().__init__()
        gates = []
        dim = 2 ** n_qubit
        for i in range(n_qubit):
            c_idx, v_idx = i, (i + 1) % n_qubit
            g = torch.eye(dim)
            for j in range(dim):
                if (j // 2 ** c_idx) % 2 == 1 and (j // 2 ** v_idx) % 2 == 1:
                    g[j, j] = -1            
            gates.append(g)

        self.op = nn.Parameter(reduce(lambda x, y: x * y, gates), requires_grad=False)

    def forward(self, x: torch.Tensor):
        return (x[0] @ self.op, x[1] @ self.op)


class Exp(nn.Module):

    def __init__(self, n_qubit: int):
        super().__init__()
        # comments shape for n_qubit = 3
        r = torch.arange(2 ** n_qubit).long()
        self.m = self.binary(r, n_qubit).float()  # (8, 3)

    def forward(self, x):
        return (torch.abs(x) ** 2) @ self.m

    def binary(self, x, bits):
        mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()