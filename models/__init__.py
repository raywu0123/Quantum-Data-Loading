from .gan import GAN
from .qgan import QGAN
from .vae import VAE
from .qae import QAE
from .qcbm import QCBM
from .ddqcl import DDQCL


MODEL_HUB = {
    "gan": GAN,
    "qgan": QGAN,
    'vae': VAE,
    'qae': QAE,
    'qcbm': QCBM,
    'ddqcl': DDQCL
}
