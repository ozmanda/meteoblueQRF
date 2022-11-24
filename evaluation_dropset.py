import os
import joblib
import csv
import numpy as np
import scipy as sp
import pandas as pd
from utils import start_timer, end_timer
import seaborn as sns
from pandas import DataFrame
from datetime import datetime
import matplotlib.pyplot as plt
from argparse import ArgumentParser


def empty_dict(keylist):
    empty_dict = {}
    for key in keylist:
        empty_dict[key] = []
    return empty_dict


def empty_df(keylist):
    empty_df = {}
    for key in keylist:
        empty_df[key] = []
    return DataFrame(empty_df)


def gather_data(datapath, featurepath):
    '''
    Function combines the dropset error predictions and the feature values to one dataset
    '''
    datapath = f'{datapath}/errors'
    savepath = f'{datapath}/dropset_data.z'
    if not os.path.isfile(savepath):
        data = {}
        for file in os.listdir(featurepath):
            stationname = file.split('.')[0]
            features = pd.read_csv(f'{featurepath}/{file}', delimiter=';')
            try:
                dropseterrors = pd.read_csv(f'{datapath}/errors_{stationname}.csv', delimiter=',')
            except FileNotFoundError:
                print(f'No file found for station {stationname}')
                continue

            # rename rows in feature dataset to datetime and remove datetime column from dropseterror dataframe
            features.rename(index=lambda i: features.loc[i]['datetime'])
            dropseterrors.rename(index=lambda i: dropseterrors.loc[i]['Datetime'])
            dropseterrors.drop('Datetime', inplace=True, axis=1)

            # renamed rows allow us to only join rows which exist in both datasets
            stationdata = pd.concat([dropseterrors, features], axis=1, join='inner')
            data[stationname] = stationdata
        joblib.dump(data, savepath, compress=3)
    else:
        data = joblib.load(savepath)

    return data


def training_data(stationdata, station):
    # initialise dictionary
    keylist = stationdata[station].keys()
    training_data = empty_dict(keylist)

    # add data from each station used during training
    for stationname in stationdata.keys():
        if stationname == station:
            continue
        for key in keylist:
            training_data[key].extend(stationdata[stationname][key])

    return DataFrame(training_data)


def data_preprocessing(stationdata, stationname, featurenames):
    trainingdata = training_data(stationdata, stationname)
    stationdata = stationdata[stationname]

    # round float features (all of them)
    trainingdata[featurenames] = trainingdata[featurenames].apply(lambda row: round(row, 4), axis=0)
    stationdata[featurenames] = stationdata[featurenames].apply(lambda row: round(row, 4), axis=0)

    return trainingdata, stationdata


def extrapolated(featurename, testdata, trainingdata):
    min025 = np.min(trainingdata[featurename])
    max975 = np.max(trainingdata[featurename])
    # min025 = trainingdata[featurename][round(len(trainingdata[featurename])*0.025)]
    # max975 = trainingdata[featurename][round(len(trainingdata[featurename])*0.975)]

    extrapolated_data = empty_df(testdata.keys())
    idxs = []
    for idx, row in testdata.iterrows():
        if row[featurename] < min025 or max975 < row[featurename]:
            idxs.append(idx)
    extrapolated_data = pd.concat([extrapolated_data, testdata.loc[idxs]])

    return extrapolated_data, idxs


def interpolated(featurename, testdata, trainingdata, extrapolation_idxs, dev=0.2):
    interpolated_data = empty_df(testdata.keys())
    eval_idxs = [idx for idx in testdata.index if idx not in extrapolation_idxs]
    if not eval_idxs:
        return interpolated_data
    else:
        idxs = []
        for idx in eval_idxs:
            for val in trainingdata[featurename]:
                intp = True
                if val-testdata.iloc[idx][featurename] <= dev:
                    intp = False
                    break
            if intp:
                idxs.append(idx)

        interpolated_data = pd.concat([interpolated_data, testdata.loc[idxs]])
        return interpolated_data


def interpolation_extrapolation(data, featurenames):
    for stationname in data.keys():
        analysed_stations = ['C0B425A3EF9E']
        if stationname in analysed_stations:
            continue
        print(f'Analysing station {stationname}......')
        trainingdata, stationdata = data_preprocessing(data, stationname, featurenames)

        for feature in featurenames:
            print(f'    Feature {feature}................')
            extrapolated_data, extrapolated_idxs = extrapolated(feature, stationdata, trainingdata)
            interpolated_data = interpolated(feature, stationdata, trainingdata, extrapolated_idxs)

            if not len(extrapolated_data) and not len(interpolated_data):
                print(f'        values covered by training set')
            else:
                fig, ax = plt.subplots()
                ax.scatter(trainingdata[feature], trainingdata['Absolute Deviation'], c='black', label=f'Training data')
                if len(extrapolated_data):
                    ax.scatter(extrapolated_data[feature], extrapolated_data['Absolute Deviation'], c='red',
                               label='Extrapolated data', s=1, marker='x')
                if len(interpolated_data):
                    ax.scatter(interpolated_data[feature], interpolated_data['Absolute Deviation'], c='blue',
                               label='Interpolated data', s=1, marker='x')

                figurepath = f'Data/DropsetData/Graphs/{stationname}_{feature}.png'
                ax.set_ylabel('Absolute Deviation [°C]')
                ax.set_xlabel(f'{feature}')
                ax.legend()
                ax.set_title(f'{stationname} {feature} inter- and extrapolation errors vs. training data', wrap=True)
                plt.savefig(figurepath)


def analyse_errors():
    datapath = 'Data/DropsetData'
    featurepath = 'Data/MeasurementFeatures_v6'
    # featurenames = ['urbangreen', 'urbangreen_10',
    #                 'urbangreen_30', 'urbangreen_100', 'urbangreen_200', 'urbangreen_500', 'humidity', 'irradiation',
    #                 'moving_average', 'temperature']
    featurenames = ['altitude', 'buildings_10', 'buildings_30', 'buildings_100', 'buildings_200',
                    'buildings_500', 'forests', 'forests_10', 'forests_30', 'forests_100', 'forests_200',
                    'forests_500', 'pavedsurfaces', 'pavedsurfaces_10', 'pavedsurfaces_30', 'pavedsurfaces_100',
                    'pavedsurfaces_200', 'pavedsurfaces_500', 'surfacewater', 'surfacewater_10', 'surfacewater_30',
                    'surfacewater_100', 'surfacewater_200', 'surfacewater_500', 'urbangreen', 'urbangreen_10',
                    'urbangreen_30', 'urbangreen_100', 'urbangreen_200', 'urbangreen_500', 'humidity', 'irradiation',
                    'moving_average', 'temperature']
    data = gather_data(datapath, featurepath)
    interpolation_extrapolation(data, featurenames)


if __name__ == '__main__':
    analyse_errors()




    # gather_stats(datapath)

