from pprint import pprint

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

from data import DATA_HUB
from models import MODEL_HUB
from models.utils import counts
from parser import get_parser
from utils import ints_to_bits


def get_pmf(counts: np.array):
    return counts / np.sum(counts)


def evaluate(target: np.array, outcome: np.array):
    assert(target.shape == outcome.shape)
    target_pmf = get_pmf(target)
    outcome_pmf = get_pmf(outcome)
    
    print(outcome_pmf)
    print(target_pmf)
    return {
        'kl': entropy(target_pmf, outcome_pmf),
        'js': jensenshannon(target_pmf, outcome_pmf) ** 2,
    }


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    data = DATA_HUB[args.data](range_=2 ** args.n_qubit)
    data_points = data.get_data(num=args.N)

    model = MODEL_HUB[args.model](
        n_qubit=args.n_qubit,
        batch_size=args.batch_size,
        n_epoch=args.n_epoch,
    )
    outcome = model.fit(data_points)

    data_counts = counts(data_points, args.n_qubit)
    eval_result = evaluate(data_counts, outcome)
    pprint(eval_result)
