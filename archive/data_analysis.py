import argparse
import os
import numpy as np
import pandas as pd
import seaborn as sns
from warnings import warn
from collections import Counter
import matplotlib.pyplot as plt
from qrf_utils import empty_df, data_in_window
# from statistical_analysis import add_month_season

# GLOBAL VARIABLES
seasons = ['summer', 'autumn', 'winter', 'spring']
geospatial_features = ['altitude', 'buildings', 'buildings_10', 'buildings_30', 'buildings_100', 'buildings_200',
                       'buildings_500', 'forests', 'forests_10', 'forests_30', 'forests_100', 'forests_200',
                       'forests_500', 'pavedsurfaces', 'pavedsurfaces_10', 'pavedsurfaces_30', 'pavedsurfaces_100',
                       'pavedsurfaces_200', 'pavedsurfaces_500', 'surfacewater', 'surfacewater_10',
                       'surfacewater_30', 'surfacewater_100', 'surfacewater_200', 'surfacewater_500', 'urbangreen',
                       'urbangreen_10', 'urbangreen_30', 'urbangreen_100', 'urbangreen_200', 'urbangreen_500']
meteorological_features = ['humidity', 'irradiation', 'moving_average', 'temperature']
error_features = ['True Temperature', 'Predicted Temperature', '2.5%', '97.5%', 'Absolute Deviation']
seasonfeatures = ['humidity', 'irradiation', 'moving_average', 'temperature', 'True Temperature',
                  'Predicted Temperature', 'Absolute Deviation']


# DATA PREPARATION
def gather_data(datapath):
    data = {}
    for filename in os.listdir(datapath):
        stationname = filename.split('.')[0]
        try:
            file = pd.read_csv(os.path.join(datapath, filename), delimiter=';')
        except FileNotFoundError:
            print(f'No file found for station {stationname}')
            continue

        try:
            file['datetime'] = pd.to_datetime(file['datetime'])
        except KeyError:
            file = pd.read_csv(os.path.join(datapath, filename), delimiter=',')
            file['datetime'] = pd.to_datetime(file['datetime'])

        data[stationname] = file

    return data


def season_classification(dt):
    month = pd.to_datetime(dt).month
    if month in [6, 7, 8]:
        return 'summer'
    elif month in [9, 10, 11]:
        return 'autumn'
    elif month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    else:
        warn(f'Month could not be identified, check datetime and try again ({month})')
        raise ValueError


def add_month_season(data):
    '''
    Function adds month and season + station feature to the data. Input is a dictionary of Dataframes, output is
    one singular dataframe. The station id is transferred from key to column variable.
    '''
    full_data = []
    for key in data.keys():
        data[key]['month'] = pd.to_datetime(data[key]['datetime']).dt.strftime('%b')
        data[key]['season'] = data[key]['datetime'].map(season_classification)
        data[key]['station'] = key
        if len(full_data) == 0:
            full_data = empty_df(data[key].keys())
        full_data = pd.concat([full_data, data[key]], axis=0, join='outer')

    return full_data


# DATA DISTRIBUTIONS
def season_distribution(data, statpath):
    # analyse amount of data per season
    # ax = sns.histplot(data, x='season', color='season', stat='density')
    ax = sns.histplot(data, x='season', stat='density')
    ax.set(xlabel='Season', ylabel='Density', title=f'Observations per season')
    plt.savefig(os.path.join(statpath, 'season_hist.png'))
    plt.close()
    monthcounts = Counter(data['month'])
    print(f'Month counts: \n{monthcounts}')


def month_distribution(data, statpath):
    # ax = sns.histplot(data, x='month', color='month', stat='density')
    ax = sns.histplot(data, x='month', stat='density')
    ax.set(xlabel='Month', ylabel='Density', title=f'Observations per month')
    plt.savefig(os.path.join(statpath, 'month_hist.png'))
    plt.close()


def hour_distribution(data, statpath):
    # ax = sns.histplot(data, x='time', col='month', stat='density')
    ax = sns.histplot(data, x='time', stat='density')
    ax.set(xlabel='Time', ylabel='Density', title=f'Observations per hour')
    plt.savefig(os.path.join(statpath, 'hour_hist.png'))
    plt.close()


# FEATURE DISTRIBUTIONS
def features_by_season(df, statpath):
    for feature in seasonfeatures:
        ax = sns.boxplot(x='season', y=feature, data=df)
        ax.set(ylabel='Season', xlabel=feature, title=f'{feature} values by season')
        plt.savefig(os.path.join(statpath, f'{feature}_season_distribution.png'))
        plt.close()


def features_by_month(df, statpath):
    for feature in seasonfeatures:
        ax = sns.boxplot(x='month', y=feature, data=df)
        ax.set(xlabel='Month', ylabel=f'{feature} value', title=f'Monthly distribution of {feature}')
        plt.savefig(os.path.join(statpath, f'{feature}_monthly_distribution.png'))
        plt.close()


def features_by_time(df, statpath):
    for feature in seasonfeatures:
        ax = sns.boxplot(x='time', y=feature, data=df)
        ax.set(xlabel='Time', ylabel=f'{feature} value', title=f'Hourly distribution of {feature}')
        plt.savefig(os.path.join(statpath, f'{feature}_season_distribution.png'))
        plt.close()


def feature_hist(df, statpath):
    features = meteorological_features
    features.extend(geospatial_features)
    for feature in features:
        try:
            ax = sns.histplot(df[feature], stat='density')
            ax.set(xlabel=f'{feature} value', ylabel='Density', title=f'{feature} density distribution')
            plt.savefig(os.path.join(statpath, f'{feature}_hist.png'))
            plt.close()
        except ValueError:
            print(f'plot for feature {feature} failed')
            pass


def analyse_feature_distributions(data , statpath):
    feature_hist(data, statpath)
    features_by_month(data, statpath)
    features_by_season(data, statpath)
    features_by_time(data, statpath)


def analyse_data_distribution(data, statpath):
    # analyse observations per season and month
    season_distribution(data, statpath)
    month_distribution(data, statpath)
    hour_distribution(data, statpath)


def data_prep(featurepath):
    data = gather_data(featurepath)
    data = add_month_season(data)
    data['datetime'] = pd.to_datetime(data['datetime'])
    data['hour'] = pd.to_datetime(data['time']).dt.hour
    return data


def statistical_analysis(datapath, statpath):
    data = data_prep(datapath)
    analyse_data_distribution(data, statpath)
    analyse_feature_distributions(data, statpath)


def station_analysis(ts_path, datapath, tstype, timeframe=None, suf=None):
    if not os.path.isdir(ts_path):
        os.mkdir(ts_path)
    for stationfile in os.listdir(datapath):
        stationname = stationfile.split('.csv')[0]
        file = pd.read_csv(os.path.join(datapath, stationfile), delimiter=';')
        file['datetime'] = pd.to_datetime(file['datetime'])
        if timeframe:
            file, _ = data_in_window([timeframe[0]], [timeframe[1]], file, os.path.basename(datapath))
            if not len(file):
                continue
            suf = f'\nfrom {timeframe[0].split("_")[0]} {timeframe[0].split("_")[1]} to ' \
                  f'{timeframe[1].split("_")[0]} {timeframe[1].split("_")[1]}'

        if tstype == 'scatter':
            ax = sns.scatterplot(x='datetime', y='temperature', data=file)
        else:
            ax = sns.lineplot(x='datetime', y='temperature', data=file)

        title = f'Measured temperature at station {stationname}{suf if timeframe else ""}'
        ax.set(xlabel='Time', ylabel=f'Temperature [°C]')
        ax.set_title(title, wrap=True)
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(ts_path, f'{stationname}_temp.png'), bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', default=None, help='Relative path to the directory containing station datasets')
    parser.add_argument('--statistics', type=eval, choices=[True, False], help='Boolean value for dataset statistics')
    parser.add_argument('--stationtimeseries', default=True, type=bool, help='Boolean value for station timeseries')
    parser.add_argument('--timeseriestype', default='scatter', help='Type of graph to generate (line or scatter)')
    parser.add_argument('--savedir', default='Data/Statistics/Datasets', help='Relative path to save directory')
    parser.add_argument('--timeframe', default=None, nargs=2, help='Provide timeframe for specific timeseries in the '
                                                                   'format YYYY/MM/DD_HH:MM')
    args = parser.parse_args()

    assert args.datapath, 'Path to dataset must be given'
    assert os.path.isdir(args.datapath), 'Path to dataset not found'

    savedir = os.path.join(args.savedir, args.datapath.split("/")[-1])
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    if args.statistics:
        statpath = os.path.join(savedir, 'General_Statistics')
        if not os.path.isdir(statpath):
            os.mkdir(statpath)
        statistical_analysis(args.datapath, statpath)

    if args.stationtimeseries:
        assert args.timeseriestype == 'scatter' or args.timeseriestype == 'line', 'Only scatter and line timeseries ' \
                                                                                  'graphs are supported'
        if args.timeframe:
            starttime = args.timeframe[0].replace(':', '-').replace('/', '-')
            endtime = args.timeframe[1].replace(':', '-').replace('/', '-')
            timeseries_path = os.path.join(savedir, f'Station_Timeseries_{args.timeseriestype}_{starttime}-'
                                                    f'{endtime}')
            station_analysis(timeseries_path, args.datapath, args.timeseriestype, timeframe=args.timeframe)
        else:
            timeseries_path = os.path.join(savedir, f'Station_Timeseries_{args.timeseriestype}')
            station_analysis(timeseries_path, args.datapath, args.timeseriestype)
