
from pprint import pprint
import contextlib
import sys
from typing import List

import os
os.environ['PYTHONHASHSEED'] = str(0)
import random
random.seed(0)

import numpy as np
np.random.seed(0)
import torch
torch.random.manual_seed(0)

from matplotlib import pyplot as plt
from tqdm import tqdm

from data import DATA_HUB
from models import MODEL_HUB
from models.utils import counts
from argparser import get_parser
from utils import evaluate



class DummyFile(object):
    def write(self, x): pass


@contextlib.contextmanager
def verbose_manager(verbose: bool):
    if verbose:
        yield
    else:
        save_stdout = sys.stdout
        sys.stdout = DummyFile()
        yield
        sys.stdout = save_stdout


def calculate_eval_stats(results: List[dict]):
    if len(results) == 0:
        raise ValueError("Empty results list")

    keys = list(results[0].keys())
    soa = {k: [r[k] for r in results] for k in keys}

    mean = {k: np.mean(soa[k]) for k in keys}
    std = {f'{k}_std': np.std(soa[k], ddof=1) for k in keys}
    maximum = {f'{k}_max': np.max(soa[k]) for k in keys}
    minumum = {f'{k}_min': np.min(soa[k]) for k in keys}
    return {**mean, **std, **maximum, **minumum}


if __name__ == '__main__':
    p = get_parser()
    args = p.parse_args()

    data = DATA_HUB[args.data]()
    data_points = data.get_data(num=args.N)

    if args.show_data_hist:
        plt.hist(data_points, bins= 2 ** data.n_bit, range=(0, 2 ** data.n_bit - 1))
        plt.show()

    data_counts = counts(data_points, data.n_bit)

    iterator = range(args.repeat)
    if args.repeat > 1:
        iterator = tqdm(iterator)
    
    eval_results = []
    for _ in iterator:
        with verbose_manager(args.repeat == 1):
            model = MODEL_HUB[args.model](
                n_qubit=data.n_bit,
                batch_size=args.batch_size,
                n_epoch=args.n_epoch,
                circuit_depth=args.circuit_depth,
            )
            outcome = model.fit(data_points)

            eval_result = evaluate(data_counts, outcome)
            eval_results.append(eval_result)
            pprint(eval_result)

    eval_stats = calculate_eval_stats(eval_results)
    pprint(eval_stats)