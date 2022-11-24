import os
import time
import numpy as np
import pandas as pd
import logging


def load_data(datapath, startDatetime = None, endDatetime = None, dropset=False):
    # debug configuration
    logging.basicConfig(filename='stationdata.log', level=logging.DEBUG, filemode='w')
    noData = []

    # For Dropset: dictionary with station name as key and pandas DataFrame as value
    # Not Dropset: concatenate DataFrames
    if dropset:
        datasets = {}

    for idx, filename in enumerate(os.listdir(datapath)):
        file = pd.read_csv(os.path.join(datapath, filename), delimiter=';')
        if idx == 0 and not dropset:
            data = pd.DataFrame(columns=file.columns)

        if startDatetime and endDatetime:
            # extract data points within the time window, if one is given
            if len(startDatetime) > 1:
                inWindow = [False for x in range(len(file))]
                for idx, startTime in enumerate(startDatetime):
                    afterStart = (pd.to_datetime(file['datetime']) >= pd.to_datetime(startTime, format='%Y/%m/%d_%H:%M')).to_list()
                    beforeEnd = (pd.to_datetime(file['datetime']) <= pd.to_datetime(endDatetime[idx], format='%Y/%m/%d_%H:%M')).to_list()
                    inWindow_ = [a and b for a, b in zip(afterStart, beforeEnd)]

                    # if station has no datapoints within time window, output to logfile and continue to next station
                    if np.sum(inWindow_) == 0:
                        noData.append(filename.split(".csv")[0])
                        continue
                    else:
                        inWindow = [a or b for a, b in zip(inWindow, inWindow_)]

                file = file.iloc[file.index[inWindow]]

            elif len(startDatetime) == 1:
                afterStart = (pd.to_datetime(file['datetime']) >= pd.to_datetime(startDatetime[0], format='%Y/%m/%d_%H:%M')).to_list()
                beforeEnd = (pd.to_datetime(file['datetime']) <= pd.to_datetime(endDatetime[0], format='%Y/%m/%d_%H:%M')).to_list()
                inWindow = file.index[[a and b for a, b in zip(afterStart, beforeEnd)]]

                # if station has no datapoints within time window, output to logfile and continue to next station
                if len(inWindow) == 0:
                    noData.append(filename.split(".csv")[0])
                    continue
                # isolate relevant features and append to relevant dataset depending on dropset True/False
                file = file.iloc[inWindow]

        if not dropset:
            data = pd.concat((data, file))
        elif dropset:
            # add rows to datasets dictionary for Dropset
            datasets[f'{filename.split(".csv")[0]}'] = file.dropna()

    logging.debug(f'{len(noData)} stations of {len(os.listdir(datapath))} have no data within the given time period')
    logging.debug(f'List of stations with no data:\n{noData}')

    if dropset:
        return datasets
    elif not dropset:
        return data


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
