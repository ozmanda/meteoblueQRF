import argparse
import os
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
from utils import empty_df, empty_dict
from pandas import DataFrame
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


# +-----------------------+
# | OUTPUT ERROR ANALYSIS |
# +-----------------------+
def split_predictions(predictions):
    preds = []
    for pred in predictions:
        pred_split = pred.split('[')[1].split(']')[0].split(' ')
        pred_split = list(filter(None, pred_split))
        pred_split = [eval(i) for i in pred_split]
        preds.append(pred_split)
    return preds


def normalise_datetime(dt):
    timenow = datetime.now()
    dt = dt.replace(year=timenow.year, month=timenow.month, day=timenow.day)
    return dt


def load_data(datapath):
    try:
        data = pd.read_csv(datapath, delimiter=',')
    except FileNotFoundError:
        print(f'No file found for {datapath.split("/")[-1]}')
        raise FileNotFoundError

    try:
        data['datetime'] = pd.to_datetime(data['datetime'])
    except KeyError:
        data = pd.read_csv(datapath, delimiter=';')
        data['datetime'] = pd.to_datetime(data['datetime'])

    data[['2.5%', 'Predicted Temperature', '97.5%']] = split_predictions(data['Prediction'])
    data['Deviation'] = np.round(data['True Temperature'] - data['Predicted Temperature'], 2)
    return data


def error_distributions(file, savedir):
    # error distribution by true temperature, skip if already generated
    filepath = os.path.join(savedir, f'error_by_temperature.png')
    if not os.path.isfile(filepath):
        plt.figure()
        fig = sns.scatterplot(x='True Temperature', y='Deviation', data=file)
        plt.savefig(filepath)
        plt.close()

    # error distribution by datetime
    filepath = os.path.join(savedir, f'error_by_datetime.png')
    if not os.path.isfile(filepath):
        plt.figure()
        fig = sns.scatterplot(x='datetime', y='Deviation', data=file)
        plt.savefig(filepath)
        plt.close()

    # true vs. predicted temperature
    filepath = os.path.join(savedir, f'true_vs_predicted.png')
    if not os.path.isfile(filepath):
        plt.figure()
        fig = sns.scatterplot(x='True Temperature', y='Predicted Temperature', data=file)
        plt.savefig(filepath)
        plt.close()

    # error distribution by time
    filepath = os.path.join(savedir, f'error_vs_time.png')
    if not os.path.isfile(filepath):
        file['time'] = file['datetime']
        file['time'] = file['time'].map(normalise_datetime)
        plt.figure()
        fig = sns.scatterplot(x='time', y='Deviation', data=file)
        plt.savefig(os.path.join(savedir, f'error_by_time.png'))
        plt.close()


def error_by_feature(file, savedir):
    features = geospatial_features
    features.extend(['humidity', 'irradiation'])
    for feature in features:
        # case to skip if already generated
        filepath = os.path.join(savedir, f'error_by_{feature}.png')
        if os.path.isfile(filepath):
            continue
        # generate and save graph
        plt.figure()
        fig = sns.scatterplot(x=feature, y='Deviation', data=file)
        plt.savefig(filepath)
        plt.close()


def metrics(data, savedir, output=False):
    metrics = {}
    n = len(data)

    # catch RuntimeWarning
    warnings.filterwarnings('error', category=RuntimeWarning)
    # RMSE calculation
    metrics['RMSE'] = np.sqrt(np.sum((data['Predicted Temperature'] - data['True Temperature'])**2/n))

    # Mean
    mu_true = np.mean(data['True Temperature'])
    mu_pred = np.mean(data['Predicted Temperature'])
    metrics['True Mean'] = mu_true
    metrics['Predicted Mean'] = mu_pred

    # Standard deviation
    sd_true = np.sqrt(np.sum((data['True Temperature'] - mu_true)**2) / n)
    sd_pred = np.sqrt(np.sum((data['Predicted Temperature'] - mu_pred)**2) / n)
    metrics['True SD'] = sd_true
    metrics['Predicted SD'] = sd_pred

    # output metrics to .txt file
    write_metrics(metrics, savedir)
    if output:
        return metrics


def write_metrics(metrics, savedir):
    # create lines for .txt file
    lines = []
    for metric in metrics.keys():
        lines.append(f'{metric}:\t {metrics[metric]}\n')

    # write lines to .txt file
    with open(os.path.join(savedir, 'metrics.txt'), 'w') as file:
        file.writelines(lines)


def analyse_errors(errordata, savedir, output=False):
    # graphs
    error_distributions(errordata, savedir)
    # numerical statistics
    if output:
        return metrics(errordata, savedir, output)
    else:
        metrics(errordata, savedir, output)


# +-----------------------------------------+
# | INTER- AND EXTRAPOLATION ERROR ANALYSIS |
# +-----------------------------------------+
def gather_dropsetdata(datapath, featurepath):
    '''
    Function combines the dropset error predictions and the feature values to one dataset and returns it
    '''
    data = {}
    for file in os.listdir(featurepath):
        stationname = file.split('.')[0]
        features = pd.read_csv(f'{featurepath}/{file}', delimiter=';', parse_dates=['datetime'])
        try:
            dropseterrors = pd.read_csv(f'{datapath}/errors_{stationname}.csv', delimiter=',', parse_dates=['datetime'])
        except FileNotFoundError:
            print(f'No file found for station {stationname}')
            continue

        # rename rows in feature dataset to datetime and remove datetime column from dropseterror dataframe
        features.rename(index=lambda i: features.loc[i]['datetime'])
        dropseterrors.rename(index=lambda i: dropseterrors.loc[i]['datetime'])
        dropseterrors.drop('datetime', inplace=True, axis=1)

        # renamed rows allow us to only join rows which exist in both datasets
        stationdata = pd.concat([dropseterrors, features], axis=1, join='inner')
        data[stationname] = stationdata
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


def int_ext_bystation(data, featurenames, savedir):
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

                figurepath = os.path.join(savedir, f'{stationname}_{feature}.png')
                ax.set_ylabel('Absolute Deviation [°C]')
                ax.set_xlabel(f'{feature}')
                ax.legend()
                ax.set_title(f'{stationname} {feature} inter- and extrapolation errors vs. training data', wrap=True)
                plt.savefig(figurepath)


def int_ext_errors(data, savedir):
    features = geospatial_features
    features.extend(meteorological_features)
    int_ext_bystation(data, features, savedir)


# +-------------------+
# | WRAPPER FUNCTIONS |
# +-------------------+
def test_statistics(datapath, savedir):
    data = load_data(datapath)
    analyse_errors(data, savedir)


def dropset_statistics(datapath, featurepath, savedir):
    data = gather_dropsetdata(datapath, featurepath)
    rmses = {}
    # inter- and extrapolation errors
    # int_ext_errors(data, savedir)
    # per station analysis
    for stationname in data.keys():
        stationpath = f'{savedir}/{stationname}'
        if not os.path.isdir(stationpath):
            os.mkdir(stationpath)
        # error_by_feature(data[stationname], stationpath)
        metrics = analyse_errors(data[stationname], stationpath, output=True)
        rmses[stationname] = metrics['RMSE']

    # nbins = np.round((np.max(rmses.values()) - np.min(rmses.values())))
    rmses_vals = list(rmses.values())
    rmses_vals.sort()
    plt.figure()
    fig = sns.histplot(rmses_vals, stat='frequency', bins=30)
    plt.savefig(os.path.join(savedir, 'error_frequency'))
    plt.close()

    # write txt file
    lines = []
    lines.append(f'Mean RMSE over all stations: {np.mean(rmses_vals)}\n')
    min_stat = min(rmses, key=rmses.get)
    max_stat = max(rmses, key=rmses.get)
    lines.append(f'Minimum RMSE: {rmses[min_stat]} ({min_stat})\n')
    lines.append(f'Maximum RMSE: {rmses[max_stat]} ({max_stat})\n')

    min05 = rmses_vals[round(len(rmses_vals)*0.05)]
    max95 = rmses_vals[round(len(rmses_vals)*0.95)]

    lines.append('Lowest and Highest 5% of RMSEs:\n')
    lowest = []
    for key in rmses.keys():
        if rmses[key] <= min05:
            lines.append(f'{key}:\t {rmses[key]}\n')
    for key in rmses.keys():
        if rmses[key] >= max95:
            lines.append(f'{key}:\t {rmses[key]}\n')

    with open(os.path.join(savedir, 'dropset_overview.txt'), 'w') as file:
        file.writelines(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default='test', help='Type of output to be analysed (test, dropset, inference)')
    parser.add_argument('--datapath', default=None, help='Relative path to the output data')
    parser.add_argument('--featurepath', default=None, help='Relative path to feature data (for dropset only)')
    args = parser.parse_args()

    assert args.datapath, 'A datapath must be given.'
    assert args.type in ['test', 'dropset', 'inference'], 'The type given must be either test, dropset or inference.'

    if args.type == 'test' or args.type == 'inference':
        # testing and inference require the same analysis
        assert args.datapath.endswith(".csv"), 'A relative path to a .csv file must be given.'
        foldername = args.datapath.split("/")[-1].split('.csv')[0]
        savedir = f'Data/Statistics/{"Testing" if args.type == "test" else "Inference"}/{foldername}'
        if not os.path.isdir(savedir):
            os.mkdir(savedir)
        test_statistics(args.datapath, savedir)

    elif args.type == 'dropset':
        assert os.path.isdir(args.datapath), 'The specified datapath must be a folder for dropset analysis'
        assert args.featurepath, 'Path to feature data must be given'
        savedir = f'Data/Statistics/Dropset/{args.datapath.split("/")[-1]}'
        if not os.path.isdir(savedir):
            os.mkdir(savedir)
        dropset_statistics(args.datapath, args.featurepath, savedir)
