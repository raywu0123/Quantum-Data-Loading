import numpy as np

from utils import bits_to_ints, ints_to_bits


def test_bit_int_conversion():
    n_bit = 3
    ints = np.arange(0, 2 ** n_bit).astype(int)
    bits = ints_to_bits(ints, n_bit)
    ints_ = bits_to_ints(bits, n_bit)

    np.testing.assert_equal(ints, ints_)
