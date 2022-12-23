import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistical_analysis import add_month_season
from utils import empty_df, empty_dict
from warnings import warn

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

def year(a, b):
    return a == b


def groupby_year(stationdata):
    start = stationdata['datetime'][0].year
    end = stationdata['datetime'][len(stationdata)-1].year
    years = empty_dict([year for year in range(start, end+1)])
    for year in range(start, end+1):
        years[year] =

    for row in stationdata.iterrows():
        years

    inyear = file.iloc[file['datetime'].year == year]



def timeseries_full(stationdata):
    for feature in meteorological_features:
        plt.plot('datetime', feature, data=stationdata)


def timeseries_month(stationdata):
    for month in range(1, 13):


def analyse_stations(datapath):
    for filename in os.listdir(datapath):
        try:
            stationdata = pd.read_csv(os.path.join(datapath, filename), delimiter=';', parse_dates=['datetime'])
            stationdata = add_month_season(stationdata)
        except ValueError:
            try:
                stationdata = pd.read_csv(os.path.join(datapath, filename), delimiter=',', parse_dates=['datetime'])
            except FileNotFoundError:
                warn(f'File {filename} not found.')
                continue
        except FileNotFoundError:
            warn(f'File {filename} not found.')
            continue
        timeseries_full(stationdata)


if __name__ == '__main__':
    ''' Conducts baseline analyses for data provided to a QRF training or dropset run. '''
    datapath = 'Data/MeasurementFeatures_v6'
    savepath = 'Data/Analysis'
