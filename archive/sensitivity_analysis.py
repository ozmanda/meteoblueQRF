import numpy as np
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='Path to model for sensitivity analysis', default=None)


    args = parser.parse_args()
    assert args.model, 'A model must be given for the sensitivity analysis'
