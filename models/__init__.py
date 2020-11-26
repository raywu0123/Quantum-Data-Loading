from .gan import GAN
from .qgan import QGAN
from .vae import VAE
from .qvae import QVAE
from .qae import QAE


MODEL_HUB = {
    "gan": GAN,
    "qgan": QGAN,
    'vae': VAE,
    'qvae': QVAE,
    'qae': QAE,
}
