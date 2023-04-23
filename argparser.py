from argparse import ArgumentParser

from data import DATA_HUB
from models import MODEL_HUB


def get_parser():
    p = ArgumentParser()
    p.add_argument('-d', '--data', type=str, choices=DATA_HUB.keys())
    p.add_argument('-N', type=int, default=20000, help="Number of training data")
    p.add_argument('-show_data_hist', action='store_true')

    p.add_argument('-m', '--model', type=str, choices=MODEL_HUB.keys())
    p.add_argument('-cd', '--circuit_depth', type=int, default=None)
    p.add_argument('-batch_size', type=int, default=2000)
    p.add_argument('-n_epoch', type=int, default=50)

    p.add_argument('-repeat', type=int, default=1)
    return p
