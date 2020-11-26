from .log_normal import LogNormal
from .triangular import Triangular
from .bimodal import Bimodal


DATA_HUB = {
    'log_normal': LogNormal,
    'triangular': Triangular,
    'bimodal': Bimodal,
}
