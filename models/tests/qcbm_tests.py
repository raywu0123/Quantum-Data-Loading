import torch

from ..qcbm import MMD


def test_to_binary():
    n_kernel, n_qubit = 3, 3
    mmd = MMD(n_kernel, n_qubit)
    x = torch.arange(2 ** n_qubit)
    b = mmd.to_binary(x)
    
    expected_b = torch.Tensor([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ])
    assert (b == expected_b).all()
