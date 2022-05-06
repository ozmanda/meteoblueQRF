import os
import numpy as np
import pandas as pd
from warnings import warn
import logging


def load_data(datapath, startDatetime = None, endDatetime = None):
    # debug configuration
    logging.basicConfig(filename='stationdata.log', level=logging.DEBUG)
    noData = []
    # create datasets dictionar with the name of the station (filename) as the key and pandas DataFrame as the value
    datasets = {}
    for filename in os.listdir(datapath):
        file = pd.read_csv(os.path.join(datapath, filename), delimiter=';')

        # extract data points within the time window, if one is given
        if startDatetime and endDatetime:
            afterStart = file.index[pd.to_datetime(file['datetime']) >= pd.to_datetime(startDatetime)].to_list()
            beforeEnd = file.index[pd.to_datetime(file['datetime']) <= pd.to_datetime(endDatetime)].to_list()
            inWindow = afterStart and beforeEnd

            # user a warning if station has no datapoints within time window and continue to next station
            if len(inWindow) == 0:
                noData.append(filename.split(".csv")[0])
                continue

            file = file.iloc[inWindow]
        # add the rows to the datasets dictionary
        datasets[f'{filename.split(".csv")[0]}'] = file.dropna()
    logging.debug(f'{len(noData)} stations of {len(os.listdir(datapath))} have no data within the given time period')
    logging.debug(f'List of stations with no data:\n{noData}')
    return datasets
