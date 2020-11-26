import numpy as np


epsilon = 1e-12


def bits_to_ints(bits: np.array, n_bit: int) -> np.array:
    bits = bits.astype(int)
    base = np.reshape(2 ** np.arange(0, n_bit)[::-1], [n_bit, 1]).astype(int)
    ints = (bits @ base)[:, 0]
    return ints


def ints_to_bits(ints: np.array, n_bit: int) -> np.array:
    bits = np.zeros([len(ints), n_bit])
    template = f'{{0:0{n_bit}b}}'
    for idx, i in enumerate(ints):
        s = template.format(i)
        for b in range(n_bit):
            bits[idx][b] = int(s[b])
    return bits

def ints_to_onehot(ints: np.array, num_class: int) -> np.array:
    ret = np.zeros([len(ints), num_class])
    for idx, i in enumerate(ints):
        ret[idx][i] = 1
    return ret
