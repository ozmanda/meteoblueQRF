import os
import joblib
import csv
import numpy as np
import scipy as sp
from warnings import warn
import pandas as pd
from utils import start_timer, end_timer
import seaborn as sns
from pandas import DataFrame
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from argparse import ArgumentParser
from evaluation_dropset import empty_df, empty_dict


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
def season_distribution(data):
    # analyse amount of data per season & month
    plt.hist(data['month'])
    plt.pause(0)
    monthcounts = Counter(data['month'])
    print(f'Month counts: \n{monthcounts}')


def month_distribution(data):
    ax = sns.histplot(data, x='month', color='month', stat='density')
    ax.set(xlabel='Month', ylabel='Density', title=f'Observations per month')


def hour_distribution(data):
    ax = sns.histplot(data, x='time', col='month', stat='density')
    ax.set(xlabel='Time', ylabel='Density', title=f'Observations per hour')


# FEATURE DISTRIBUTIONS
def features_by_season(df):
    for feature in seasonfeatures:
        ax = sns.boxplot(x='season', y=feature, data=df)
        ax.set(ylabel='Season', xlabel=feature, title=f'{feature} values by season')
        plt.show()
        plt.pause(0)


def features_by_month(df):
    for feature in seasonfeatures:
        ax = sns.boxplot(x='month', y=feature, data=df)
        ax.set(xlabel='Month', ylabel=f'{feature} value', title=f'Monthly distribution of {feature}')
        plt.show()
        plt.pause(0)


def features_by_time(df):
    for feature in seasonfeatures:
        ax = sns.boxplot(x='month', y=feature, data=df)
        ax.set(xlabel='Month', ylabel=f'{feature} value', title=f'Monthly distribution of {feature}')
        plt.show()
        plt.pause(0)


def feature_hist(df):
    features = meteorological_features
    features.extend(geospatial_features)
    for feature in features:
        try:
            ax = sns.histplot(df[feature], stat='density')
            ax.set(xlabel=f'{feature} value', ylabel='Density', title=f'{feature} density distribution')
            plt.show()
            plt.pause(0)
        except ValueError:
            print(f'plot forfeature {feature} failed')
            pass


# ERROR DISTRIBUTIONS
def error_vs_season(data):
    ax = sns.boxplot(x='season', y='Absolute Deviation', data=data)
    ax.set(ylabel='Absolute Deviation [°C]', xlabel='Season', title='Absolute deviation by season')
    plt.show()
    plt.pause(0)


def error_vs_features(data):
    # features = meteorological_features
    # features.extend(geospatial_features)
    for feature in meteorological_features:
        ax = sns.scatterplot(x=feature, y='Absolute Deviation', data=data)
        ax.set(ylabel='Absolute Deviation [°C]', xlabel=feature, title=f'Absolute deviation by {feature}')
        plt.show()
        plt.pause(0)


def errors_vs_month(data):
    ax = sns.boxplot(x='month', y='Absolute Deviation', data=data)
    ax.set(ylabel='Absolute Deviation [°C]', xlabel='Month', title='Absolute deviation by month')
    plt.show()
    plt.pause(0)


# ANALYSES
def analyse_errors(data):
    # error_vs_season(data)
    error_vs_features(data)
    errors_vs_month(data)


def analyse_feature_distributions(data):
    feature_hist(data)
    features_by_month(data)
    features_by_season(data)
    features_by_time(data)


def analyse_data_distribution(data):
    # analyse observations per season and month
    season_distribution(data)
    month_distribution(data)
    hour_distribution(data)


def data_prep(featurepath):
    data = gather_data(featurepath)
    data = add_month_season(data)
    data['datetime'] = pd.to_datetime(data['datetime'])
    data['hour'] = pd.to_datetime(data['time']).dt.hour
    return data


def statistical_analysis():
    datapath = 'Data/DropsetData'
    featurepath = 'Data/MeasurementFeatures_v6'
    # savepath = 'Data/DropsetData/analysis'
    data = data_prep(featurepath)

    analyse_data_distribution(data)
    analyse_feature_distributions(data)
    # analyse_errors(data)

    


if __name__ == '__main__':
    statistical_analysis()





    # g = sns.FacetGrid(data, col='month', sharex=False)
    # g.map(sns.scatterplot, 'datetime', 'Absolute Deviation')
    # xformatter = mdates.DateFormatter("%d")
    # g.axes[0, 3].xaxis.set_major_formatter(xformatter)

    # def gather_seasons(data, season_features):
    #     data = add_month_season(data)
    #     season_features.append('month')
    #     savepath = 'Data/DropsetData/analysis/'
    #     # create empty dictionary with an empty pandas DataFrame for each season (only meteorological features)
    #     season_dfs = empty_dict(seasons)
    #     for season in seasons:
    #         season_dfs[season] = empty_df(season_features)
    #
    #     for key in data.keys():
    #         season_idxs = {'summer': [], 'autumn': [], 'winter': [], 'spring': []}
    #         for idx, row in data[key].iterrows():
    #             for season in seasons:
    #                 if row['season'] == season:
    #                     season_idxs[season].append(idx)
    #                     break
    #         for season in seasons:
    #             if not season_idxs[season]:
    #                 continue
    #             season_dfs[season] = pd.concat([season_dfs[season], data[key].loc[season_idxs[season], season_features]],
    #                                         axis=0, join='outer')
    #
    #     return season_dfs
    #
    #
    # def gather_feature_errors(data):
    #     feature_errors = {}
    #     for feature in meteorological_features:
    #         fe = empty_df([feature, 'Absolute Deviation'])
    #         for key in data.keys():
    #             fe = pd.concat([fe, data[key].loc[:, [feature, 'Absolute Deviation']]])
    #
    #         feature_errors[feature] = fe
    #
    #     return feature_errors
