from argparse import ArgumentParser


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('-d', '--data', type=str)
    parser.add_argument('-N', type=int, default=20000, help="Number of training data")
    parser.add_argument('-show_data_hist', action='store_true')

    parser.add_argument('-m', '--model', type=str)
    parser.add_argument('-cd', '--circuit_depth', type=int, default=None)
    parser.add_argument('-batch_size', type=int, default=2000)
    parser.add_argument('-n_epoch', type=int, default=50)

    parser.add_argument('-repeat', type=int, default=1)
    return parser
