import os
import pickle
import time
import logging
import numpy as np
import pandas as pd
import _pickle as cPickle
import joblib
from seaborn import heatmap
import matplotlib.pyplot as plt


def empty_dict(keylist):
    empty_dict = {}
    for key in keylist:
        empty_dict[key] = []
    return empty_dict


def empty_df(keylist):
    empty_df = {}
    for key in keylist:
        empty_df[key] = []
    return pd.DataFrame(empty_df)


def initialise_empty_df(filepath, dropset=False):
    if not dropset:
        file = pd.read_csv(filepath, delimiter=';')
        if not test_data(file):
            file = pd.read_csv(filepath, delimiter=',')
        return pd.DataFrame(columns=file.columns)


def test_data(file):
    try:
        test = file['datetime']
        return True
    except KeyError:
        return False


def unravel_data(data):
    print('    unravelling data')
    unraveled = empty_df(data.keys())
    for key in data.keys():
        unraveled[key] = np.ravel(data[key])
    return unraveled, data[key].shape


def load_file(datapath):
    with open(datapath, 'rb') as file:
        data = cPickle.load(file)
        file.close()
    return data

def save_object(path, object):
    '''
    Wrapper for object saving which automatically detects whether an object being saved is a numpy array or not and
    uses the respective save method. Joblib contains special implementations for NumPy arrays and is significantly
    faster than Pickle, otherwise Pickle (which is implemented in C) is faster.
    '''
    if type(object) == np.ndarray:
        joblib.dump(object, f'{path}.json', compress=3)
    else:
        with open(f'{path}.json', 'wb') as file:
            cPickle.dump(object, file, protocol=pickle.HIGHEST_PROTOCOL)
            file.close()


def reshape_preds(preds, map_shape):
    '''
    Reshapes the predictions, which come as a list of three values [lower CI bound, mean, upper CI bound]. The resulting
    map shape is identical to the loaded maps, with the exception of containing 3-channel data.
    original map shape (10, 300, 300) --> prediction map shape (10, 300, 300, 3)
    '''
    return preds.reshape((map_shape[0], map_shape[1], map_shape[2], 3))


def load_file(datapath):
    file = pd.read_csv(os.path.join(datapath, filename), delimiter=';')
    # check correct delimiter usage (not uniform)
    if not test_data(file):
        file = pd.read_csv(os.path.join(datapath, filename), delimiter=',')
    return file


def load_data(datapath, startDatetime = None, endDatetime = None, dropset=False):
    # debug configuration
    logging.basicConfig(filename='stationdata.log', level=logging.DEBUG, filemode='w')

    # Not Dropset: concatenate DataFrames
    noData = []
    data = initialise_empty_df(os.path.join(datapath, os.listdir(datapath)[0]))
    for idx, filename in enumerate(os.listdir(datapath)):
        file = load_file(datapath)
        # extract data points within the time window, if one is given
        if startDatetime and endDatetime:
            file, noData = data_in_window(startDatetime, endDatetime, file, filename)
        data = pd.concat((data, file))
    if noData:
        logging.debug( f'{len(noData)} stations of {len(os.listdir(datapath))} have no data within the given time period')
        logging.debug(f'List of stations with no data:\n{noData}')
    return data

def load_dropset_data(datapath, startDatetime = None, endDatetime = None):
    # For Dropset: dictionary with station name as key and pandas DataFrame as value
    noData = []
    datasets = {}
    for idx, filename in enumerate(os.listdir(datapath)):
        file = load_file(datapath)
        # extract data points within the time window, if one is given
        if startDatetime and endDatetime:
            file, noData = data_in_window(startDatetime, endDatetime, file, filename)
        # add rows to datasets dictionary for Dropset
        datasets[f'{filename.split(".csv")[0]}'] = file.dropna()
    if noData:
        logging.debug( f'{len(noData)} stations of {len(os.listdir(datapath))} have no data within the given time period')
        logging.debug(f'List of stations with no data:\n{noData}')

    return datasets


def load_inference_data(datapath):
    print('Loading Data')
    tic = time.perf_counter()
    data = load_file(datapath)
    toc = time.perf_counter()
    print(f'    data loading time {toc - tic:0.2f} seconds\n')
    print('Data preprocessing')
    _ = data.pop('datetime')
    _ = data.pop('time')
    _ = data.pop('temperature')
    if 'moving average' in data.keys():
        data['moving_average'] = data['moving average']
        _ = data.pop('moving average')
    tic = time.perf_counter()
    featuremaps, mapshape = unravel_data(data)
    toc = time.perf_counter()
    print(f'    unravel time {toc - tic:0.2f} seconds\n')
    return featuremaps, mapshape


def data_in_window(starttimes, endtimes, file, filename):
    # time formatting
    starttimes = pd.to_datetime(starttimes, format='%Y/%m/%d_%H:%M')
    endtimes = pd.to_datetime(endtimes, format='%Y/%m/%d_%H:%M')
    file['datetime'] = pd.to_datetime(file['datetime'])
    noData = []
    inWindow = [False for x in range(len(file))]
    for idx, startTime in enumerate(starttimes):
        inWindow_idxs = idxs_in_window(startTime, endtimes[idx], file['datetime'])

        # if station has no datapoints within time window, output to logfile and continue to next station
        if np.sum(inWindow_idxs) == 0:
            noData.append(filename.split(".csv")[0])
            continue
        else:
            inWindow = [a or b for a, b in zip(inWindow, inWindow_idxs)]

    data = file.iloc[inWindow]
    return data, noData


def idxs_in_window(starttime, endtime, timecolumn):
    afterStart = (timecolumn >= starttime).to_list()
    beforeEnd = (timecolumn <= endtime).to_list()
    inWindow = [a and b for a, b in zip(afterStart, beforeEnd)]
    return inWindow


def map_vis(val_array: np.ndarray, savedir: str):
    '''
    Generates images of the given ndarray over all times (presumes 3 dimensions: lat, lon and time)
    '''
    if not os.path.isdir(path):
        os.mkdir(path)
    for time in range(errormap.shape[0]):
        savepath = os.path.join(path, f'errormap_t.{time}.png')
        if not os.path.isfile(savepath):
            print('    error heatmap')
            ax = heatmap(errormap[time, :, :])
            plt.show()
            plt.savefig(savepath, bbox_inches='tight')
            plt.close()


def mse(ytrue, ypred):
    dev = []
    for idx, item in enumerate(ytrue):
        dev.append((item - ypred[idx])**2)
    return np.round((1/len(ytrue)) * np.sum(dev), 4)


def sd_mu(val_list):
    n = len(val_list)
    mu = round(np.mean(val_list),2)
    sd = round(np.sqrt((np.sum(val_list-mu)**2) / (n-1)), 2)
    return sd, mu


def start_timer():
    global _start_time
    _start_time = time.time()


def end_timer():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    print(f'Time: {t_hour}:{t_min}:{t_sec}')


def timenow():
    # outputs the current time in a standardised format
    timenow = datetime.now().replace(second=0, microsecond=0)
    timenow = f'{timenow.year}-{timenow.month}-{timenow.day}_{timenow.hour}.{timenow.minute}'
    return timenow
