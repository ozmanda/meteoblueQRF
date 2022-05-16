import os
import numpy as np
import scipy as sp
import pandas as pd
from utils import mse
import seaborn as sns
from pandas import DataFrame
from datetime import datetime
import matplotlib.pyplot as plt
from argparse import ArgumentParser


def error_list(datapath, measurementpath):
    if not os.path.isfile(os.path.join(os.path.dirname(datapath), 'Station_Errors.csv')):
        features = ['altitude', 'buildings', 'buildings_10', 'buildings_30', 'buildings_100', 'buildings_200',
                    'buildings_500', 'forests', 'forests_10', 'forests_30', 'forests_100', 'forests_200', 'forests_500',
                    'pavedsurfaces', 'pavedsurfaces_10', 'pavedsurfaces_30', 'pavedsurfaces_100', 'pavedsurfaces_200',
                    'pavedsurfaces_500', 'surfacewater', 'surfacewater_10', 'surfacewater_30', 'surfacewater_100',
                    'surfacewater_200', 'surfacewater_500', 'urbangreen', 'urbangreen_10', 'urbangreen_30',
                    'urbangreen_100', 'urbangreen_200', 'urbangreen_500']
        data = []

        for filename in os.listdir(datapath):
            filedata = []
            filedata.append(filename.split('.csv')[0].split('_')[1])
            file = pd.read_csv(os.path.join(datapath, filename), delimiter=",")
            filedata.append(mse(file['True Temperature'], file['Predicted Temperature']))

            # isolate and append static features
            file = pd.read_csv(os.path.join(measurementpath, f'{filedata[0]}.csv'), delimiter=";")[features].iloc[0]
            for item in file:
                filedata.append(item)
            data.append(filedata)

        columns = ['ID', 'MSE']
        for feature in features:
            columns.append(feature)
        df = DataFrame(data, columns=columns)
        df.to_csv(os.path.join(os.path.dirname(datapath), 'Station_Errors.csv'), index=False)


def evaluate_error_list(datapath):
    file = pd.read_csv(os.path.join(os.path.dirname(datapath), 'Station_Errors.csv'), delimiter=';')
    for columnname in file.columns[2:]:
        plt.figure()
        fig = sns.scatterplot(x=columnname, y='MSE', data=file)
        plt.savefig(os.path.join(os.path.dirname(datapath), f'Error Evaluation//{columnname}_MSE.png'))
        plt.close()


def errors_by_temp(datapath):
    savedir = os.path.join(os.path.dirname(datapath), 'Temp vs. Deviation')
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    for filename in os.listdir(datapath):
        file = pd.read_csv(os.path.join(datapath, filename))
        plt.figure()
        fig = sns.scatterplot(x='True Temperature', y='Absolute Deviation', data=file)
        plt.savefig(os.path.join(savedir, f'{filename}.png'))
        plt.close()


def errors_by_datetime(datapath):
    savedir = os.path.join(os.path.dirname(datapath), 'Temp vs. Time')
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    for filename in os.listdir(datapath):
        file = pd.read_csv(os.path.join(datapath, filename))
        file['Datetime'] = pd.to_datetime(file['Datetime'])
        plt.figure()
        fig = sns.scatterplot(x='Datetime', y='Absolute Deviation', data=file)
        plt.savefig(os.path.join(savedir, f'{filename}.png'))
        plt.close()


def errors_by_time(datapath):
    savedir = os.path.join(os.path.dirname(datapath), 'Temp vs. Time TEST')
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    for filename in os.listdir(datapath):
        file = pd.read_csv(os.path.join(datapath, filename))
        time = pd.DatetimeIndex(file['Datetime']).time
        plt.figure()
        fig = sns.scatterplot(x=time, y=file['Absolute Deviation'])
        plt.savefig(os.path.join(savedir, f'{filename}.png'))
        plt.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--datafolder', help='Path to folder containing dropset data',
                        default='C://Users//ushe//PycharmProjects//pythonQRF//Data//DropsetData')
    parser.add_argument('--measurementpath', help='Path to measurement data per station',
                        default='C://Users//ushe//PycharmProjects//Stadtklima QRF//Data//MeasurementFeatures_v6')

    args = parser.parse_args()

    # evaluate_error_list(args.datafolder)
    error_list(args.datafolder, args.measurementpath)
    errors_by_time(args.datafolder)
