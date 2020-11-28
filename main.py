
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
from parser import get_parser
from utils import ints_to_bits, evaluate



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
    return {**mean, **std}

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    data = DATA_HUB[args.data](range_=2 ** args.n_qubit)
    data_points = data.get_data(num=args.N)
    data_counts = counts(data_points, args.n_qubit)


    iterator = range(args.repeat)
    if args.repeat > 1:
        iterator = tqdm(iterator)
    
    eval_results = []
    for _ in iterator:        
        with verbose_manager(args.repeat == 1):
            model = MODEL_HUB[args.model](
                n_qubit=args.n_qubit,
                batch_size=args.batch_size,
                n_epoch=args.n_epoch,
            )
            outcome = model.fit(data_points)

            eval_result = evaluate(data_counts, outcome)
            eval_results.append(eval_result)
            pprint(eval_result)

    eval_stats = calculate_eval_stats(eval_results)
    pprint(eval_stats)