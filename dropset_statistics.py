import os
import numpy as np
import pandas as pd
from qrf_utils import empty_df, empty_dict
import matplotlib.pyplot as plt

# GLOBAL VARIABLES
seasons = ['summer', 'autumn', 'winter', 'spring']
geospatial_features = ['altitude', 'buildings', 'buildings_10', 'buildings_30', 'buildings_100', 'buildings_200',
                       'buildings_500', 'forests', 'forests_10', 'forests_30', 'forests_100', 'forests_200',
                       'forests_500', 'pavedsurfaces', 'pavedsurfaces_10', 'pavedsurfaces_30', 'pavedsurfaces_100',
                       'pavedsurfaces_200', 'pavedsurfaces_500', 'surfacewater', 'surfacewater_10',
                       'surfacewater_30', 'surfacewater_100', 'surfacewater_200', 'surfacewater_500', 'urbangreen',
                       'urbangreen_10', 'urbangreen_30', 'urbangreen_100', 'urbangreen_200', 'urbangreen_500']
meteorological_features = ['humidity', 'irradiation', 'moving_average', 'temperature']

def statistics(dropsetQRF):
    """
    Calculates relevant statistics for the dropset run as a whole. Noteably, it estimates the inter- and
    extrapolation capacity of the model by analysing individual feature distributions and identifying inter- and
    extrapolation of feature/target variables.
    :return:
    """

    # Inter- and extrapolation error calculations
    int_ext_errors()


def int_ext_errors(data: dict, savedir: str):
    features = geospatial_features
    features.extend(meteorological_features)
    int_ext_bystation(data, features, savedir)


def extrapolated(featurename: str, testdata: pd.DataFrame, trainingdata: pd.DataFrame):
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


def interpolated(featurename: str, testdata: pd.DataFrame, trainingdata: pd.DataFrame,
                 extrapolation_idxs: list, dev=0.2):
    interpolated_data = empty_df(testdata.keys())
    eval_idxs = [idx for idx in testdata.index if idx not in extrapolation_idxs]
    if not eval_idxs:
        return interpolated_data
    else:
        idxs = []
        for idx in eval_idxs:
            intp = False
            for val in trainingdata[featurename]:
                intp = True
                if val-testdata.iloc[idx][featurename] <= dev:
                    intp = False
                    break
            if intp:
                idxs.append(idx)

        interpolated_data = pd.concat([interpolated_data, testdata.loc[idxs]])
        return interpolated_data


def int_ext_bystation(data: dict, featurenames: list, savedir: str):
    for stationname in data.keys():
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

                figurepath = os.path.join(savedir, f'{stationname}_{feature}.png')
                ax.set_ylabel('Absolute Deviation [°C]')
                ax.set_xlabel(f'{feature}')
                ax.legend()
                ax.set_title(f'{stationname} {feature} inter- and extrapolation errors vs. training data', wrap=True)
                plt.savefig(figurepath)


def training_data(stationdata: dict, station: str):
    # initialise dictionary
    keylist = stationdata[station].keys()
    training_data = empty_dict(keylist)

    # add data from each station used during training
    for stationname in stationdata.keys():
        if stationname == station:
            continue
        for key in keylist:
            training_data[key].extend(stationdata[stationname][key])

    return pd.DataFrame(training_data)


def data_preprocessing(stationdata: dict, stationname: str, featurenames: list):
    trainingdata = training_data(stationdata, stationname)
    stationdata = stationdata[stationname]

    # round float features (all of them)
    trainingdata[featurenames] = trainingdata[featurenames].apply(lambda row: round(row, 4), axis=0)
    stationdata[featurenames] = stationdata[featurenames].apply(lambda row: round(row, 4), axis=0)

    return trainingdata, stationdata