import os
import re
import argparse
import numpy
from seaborn import histplot, boxplot
import matplotlib.pyplot as plt
import pandas as pd
from qrf_utils import empty_df, empty_dict, sd_mu, rmse

LCZs = {'A': {'Name': 'Dense trees', 'groups': ['Land Cover']},
        'B': {'Name': 'Scattered trees', 'groups': ['Land Cover']},
        'C': {'Name': 'Bush, scrub', 'groups': ['Land Cover']},
        'D': {'Name': 'Low plants', 'groups': ['Land Cover']},
        'E': {'Name': 'Bare rock or paved', 'groups': ['Land Cover']},
        'F': {'Name': 'Bare soil or sand', 'groups': ['Land Cover']},
        'G': {'Name': 'Water', 'groups': ['Land Cover']},
        '1': {'Name': 'Compact high-rise', 'groups': ['Compact', 'Compact and Industry', 'High-rise']},
        '2': {'Name': 'Compact mid-rise', 'groups': ['Compact', 'Compact and Industry', 'Mid-rise']},
        '3': {'Name': 'Compact low-rise', 'groups': ['Compact', 'Compact and Industry', 'Low-rise']},
        '4': {'Name': 'Open high-rise', 'groups': ['Open', 'High-rise']},
        '5': {'Name': 'Open mid-rise', 'groups': ['Open', 'Mid-rise']},
        '6': {'Name': 'Open low-rise', 'groups': ['Open', 'Low-rise']},
        '7': {'Name': 'Lightweight low-rise', 'groups': ['Lightweight and Sparse', 'Low-rise', 'Lightweight']},
        '8': {'Name': 'Large low-rise', 'groups': ['Lightweight and Sparse', 'Low-rise', 'Lightweight']},
        '9': {'Name': 'Sparsely built', 'groups': ['Lightweight and Sparse', 'Low-rise']},
        '10': {'Name': 'Heavy industry', 'groups': ['Compact and Industry', 'Low-rise']}}

groups = ['Compact', 'Open', 'Lightweight', 'Land Cover', 'Compact and Industry', 'Lightweight and Sparse', 'High-rise',
          'Mid-rise', 'Low-rise']


def lcz_summary(lcz_data: dict, path: str):
    summary = empty_dict(['LCZ', 'Standard Deviation', 'Mean Error', 'RMSE'])
    for lcz in lcz_data.keys():
        true = lcz_data[lcz]['True Temperature'].to_list()
        pred = lcz_data[lcz]['Predicted Temperature'].to_list()
        if not len(true) or not len(pred):
            continue
        summary['LCZ'].append(lcz)
        sd, mu = sd_mu(lcz_data[lcz]['Deviation'])
        summary['Standard Deviation'].append(sd)
        summary['Mean Error'].append(mu)
        assert len(true) > 0, f'True data for lcz {lcz} not available'
        assert len(pred) > 0, f'Pred data for lcz {lcz} not available'
        summary['RMSE'].append(rmse(true, pred))

    summary = pd.DataFrame(summary)
    summary.to_csv(os.path.join(path, 'LCZ_summary.csv'), index=False)


def gather_errors(summary_file, path):
    """
    Generates one DataFrame per LCZ type and then concatenates according to the groups defined above: compact, open,
    lightweight, land cover, compact + industry, lightweight + sparse, high-rise, mid-rise and low-rise.
    :param summary_file: station summary file as a pandas DataFrame
    :param path: path to individual station error directory
    :return: lcz_dfs (dictionary of df with dt, true, predicted, deviation and station) and group dfs, containing the
    same information + lcz column
    """
    keys = ['datetime', 'True Temperature', 'Predicted Temperature', 'Deviation', 'Station', 'LCZ']
    lcz_dfs = empty_dict(LCZs.keys())
    group_dfs = empty_dict(groups)

    # Generate empty dataframes for each LCZ and all LCZ groupings
    for type in LCZs.keys():
        lcz_dfs[type] = empty_df(keys[:-1])
    for group in group_dfs.keys():
        group_dfs[group] = empty_df(keys)

    # add station data to corresponding LCZ and group datasets
    for idx, row in summary_file.iterrows():
        station = row['Station']
        station_data = pd.read_csv(os.path.join(path, 'errors', f'errors_{station}.csv'))
        station_data = station_data[['datetime', 'True Temperature', 'Predicted Temperature', 'Deviation']]
        station_data['station'] = [station] * len(station_data)
        for lcz in row['LCZ']:
            lcz_dfs[lcz] = pd.concat([lcz_dfs[lcz], station_data])
            lcz_dfs[lcz].reset_index()
            for group in LCZs[lcz]['groups']:
                station_data['LCZ'] = [lcz] * len(station_data)
                group_dfs[group] = pd.concat([group_dfs[group], station_data])

    lcz_summary(lcz_dfs, os.path.dirname(path))

    return lcz_dfs, group_dfs


def error_distribution(errors, path, name, group=False):
    # histplot
    imgpath = os.path.join(path, f'{name}_hist.png')
    if group:
        fig = histplot(data=errors, x='Deviation', hue='LCZ', stat='probability')
    else:
        fig = histplot(errors, stat='probability')
    fig.set_title(f'Prediction Error Distribution for {name}')
    fig.set_xlabel('Prediction Error [°C]')
    fig.set_ylabel('Probability')
    plt.savefig(imgpath)
    plt.close()

    # boxplot
    imgpath = os.path.join(path, f'{name}_box.png')
    if group:
        fig = boxplot(data=errors, y='Deviation', x='LCZ')
        fig.set_title(f'Prediction Error Distribution for {name} by Local Climate Zone')
        fig.set_xlabel('LCZ')
    else:
        fig = boxplot(errors)
        fig.set_title(f'Prediction Error Distribution for {name}')
    fig.set_ylabel('Prediction Error [°C]')
    plt.savefig(imgpath)
    plt.close()


def load_summary(datapath):
    file = pd.read_csv(datapath, delimiter=',')
    file.dropna(inplace=True)
    file.reset_index(inplace=True)
    lczs = []
    # separate LCZ classifications into a list
    for idx, row in file.iterrows():
        lcz = [*re.sub(r'[^\w]', '', row['LCZ'])]
        if 'o' in lcz: lcz.remove('o')
        if 'r' in lcz: lcz.remove('r')
        lczs.append(lcz)
    file['LCZ'] = lczs
    return file


def lcz_analysis(path):
    """
    Creates a graph of errors over time for each LCZ and group: compact, open, lightweight, land cover, compact +
    industry, lightweight + sparse, high-rise, mid-rise and low-rise.
    :param group_errors: path to individual station error directory
    :return:
    """
    summary_file = load_summary(os.path.join(path, 'station_summary.csv'))
    lcz_data, group_data = gather_errors(summary_file, path)
    imgpath = os.path.join(path, 'LCZ_analysis')
    if not os.path.isdir(path):
        os.mkdir(path)

    # graphs for LCZs individually
    for lcz in LCZs.keys():
        error_distribution(lcz_data[lcz]['Deviation'], imgpath, lcz)

    # graphs for LCZ groups
    for group in groups:
        error_distribution(group_data[group], imgpath, group, group=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datapath', type=str, help='Path to output file')
    args = parser.parse_args()

    lcz_analysis(args.datapath)
