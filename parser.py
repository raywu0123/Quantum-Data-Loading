from argparse import ArgumentParser


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('-d', '--data', type=str)
    parser.add_argument('-N', type=int, default=20000, help="Number of training data")

    parser.add_argument('-m', '--model', type=str)
    parser.add_argument('-n_qubit', type=int, default=3)
    parser.add_argument('-batch_size', type=int, default=2000)
    parser.add_argument('-n_epoch', type=int, default=50)
    return parser
