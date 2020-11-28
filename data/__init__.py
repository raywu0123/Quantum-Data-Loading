from functools import partial

from .log_normal import LogNormal
from .triangular import Triangular
from .bimodal import Bimodal
from .bar_and_stripes import BarAndStripes


DATA_HUB = {
    'log_normal': LogNormal,
    'triangular': Triangular,
    'bimodal': Bimodal,
    'bas_2x2': partial(BarAndStripes, width=2, height=2),
    'bas_3x3': partial(BarAndStripes, width=3, height=3),
}
