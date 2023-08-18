import os
import argparse
import pandas as pd
import plotly.express as px


def load_dataset(path):
    df = pd.read_csv(path, delimiter=';')
    if len(df) == 0:
        df = pd.read_csv(path, delimiter=',')
    return df


def variable_correlation(df: pd.DataFrame):
    drop_cols = []
    if drop_cols:
        df.drop(drop_cols, inplace=True)

    corr_matrix = df.corr(method='pearson')


def variable_matrix(df: pd.DataFrame):
    drop_cols = []
    if drop_cols:
        df.drop(drop_cols, inplace=True)
    fig = px.scatter_matrix()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='Path to dataset for analysis', default=None)


    args = parser.parse_args()
    assert args.dataset, 'A dataset must be given for the dataset analysis'

