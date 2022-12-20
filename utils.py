import os
import time
import numpy as np
import pandas as pd
import logging


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


def initialise_empty_df(datapath, dropset=False):
    if not dropset:
        file = pd.read_csv(os.path.join(datapath, os.listdir(datapath)[0]), delimiter=';')
        return pd.DataFrame(columns=file.columns)


def test_data(file):
    try:
        test = file['datetime']
        return True
    except KeyError:
        return False


def load_data(datapath, startDatetime = None, endDatetime = None, dropset=False):
    # debug configuration
    logging.basicConfig(filename='stationdata.log', level=logging.DEBUG, filemode='w')

    # Not Dropset: concatenate DataFrames
    if not dropset:
        noData = []
        data = initialise_empty_df(datapath)
        for idx, filename in enumerate(os.listdir(datapath)):
            file = pd.read_csv(os.path.join(datapath, filename), delimiter=';')
            # check correct delimiter usage (not uniform)
            if not test_data(file):
                file = pd.read_csv(os.path.join(datapath, filename), delimiter=',')
            # extract data points within the time window, if one is given
            if startDatetime and endDatetime:
                file, noData = data_in_window(startDatetime, endDatetime, file, filename)
            data = pd.concat((data, file))
        if noData:
            logging.debug( f'{len(noData)} stations of {len(os.listdir(datapath))} have no data within the given time period')
            logging.debug(f'List of stations with no data:\n{noData}')
        return data

    # For Dropset: dictionary with station name as key and pandas DataFrame as value
    else:
        noData = []
        datasets = {}
        for idx, filename in enumerate(os.listdir(datapath)):
            file = pd.read_csv(os.path.join(datapath, filename), delimiter=';')
            # check correct delimiter usage (not uniform)
            if not test_data(file):
                file = pd.read_csv(os.path.join(datapath, filename), delimiter=',')
            # extract data points within the time window, if one is given
            if startDatetime and endDatetime:
                file, noData = data_in_window(startDatetime, endDatetime, file, filename)
            # add rows to datasets dictionary for Dropset
            datasets[f'{filename.split(".csv")[0]}'] = file.dropna()
        if noData:
            logging.debug( f'{len(noData)} stations of {len(os.listdir(datapath))} have no data within the given time period')
            logging.debug(f'List of stations with no data:\n{noData}')

        return datasets


def data_in_window(starttimes, endtimes, file, filename):
    # time formatting
    starttimes = pd.to_datetime(starttimes, format='%Y/%m/%d_%H:%M')
    endtimes = pd.to_datetime(endtimes, format='%Y/%m/%d_%H:%M')
    file['datetime'] = pd.to_datetime(file['datetime'])
    noData = []
    inWindow = [False for x in range(len(file))]
    for idx, startTime in enumerate(starttimes):
        inWindow_idxs = idxs_in_window(startTime, endtimes[idx], file)

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


_start_time = time.time()


def mse(ytrue, ypred):
    dev = []
    for idx, item in enumerate(ytrue):
        dev.append((item - ypred[idx])**2)
    return np.round((1/len(ytrue)) * np.sum(dev), 4)


def start_timer():
    global _start_time
    _start_time = time.time()


def end_timer():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    print(f'Time: {t_hour}:{t_min}:{t_sec}')
