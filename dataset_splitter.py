import os
import re
import shutil
import numpy as np
import pandas as pd
from warnings import warn
from utils import load_data, empty_df, idxs_in_window


def normalise_umlauts(id):
    # replace ä ü ö with ae ue and oe
    try:
        id = id.replace('ä', 'ae')
        id = id.replace('ü', 'ue')
        id = id.replace('ö', 'oe')
    except AttributeError:
        pass
    return id


def remove_operators(id):
    try:
        id = re.sub('\s+', '', id)
    except TypeError:
        pass
    return id


def split_day_night(datapath):
    # define begin and end of the night
    night_start = pd.to_datetime('01:00')
    night_end = pd.to_datetime('05:00')

    # new folders for night and day datasets
    daypath = f'{datapath}_day'
    if not os.path.isdir(daypath):
        os.mkdir(daypath)
    nightpath = f'{datapath}_night'
    if not os.path.isdir(nightpath):
        os.mkdir(nightpath)

    # divide datasets and save to corresponding folder
    for filename in os.listdir(datapath):
        file = pd.read_csv(os.path.join(datapath, filename), delimiter=';')
        file['time'] = pd.to_datetime(file['time'])
        night = idxs_in_window(night_start, night_end, file['time'])
        day = [not n for n in night]

        # afterstart = (night_start <= file['time']).to_list()
        # beforeend = (night_end >= file['time']).to_list()
        # night = [a and b for a,b in zip(afterstart, beforeend)]

        night_data = file.iloc[night]
        night_data.to_csv(os.path.join(nightpath, filename), index=False)
        day_data = file.iloc[day]
        day_data.to_csv(os.path.join(daypath, filename), index=False)


def day_interval(datetimes):
    startdt = datetimes[0]
    start = [startdt.replace(hour=0, minute=0, second=0), startdt.replace(hour=23, minute=59, second=0)]
    enddt = datetimes[-1].replace(hour=23, minute=59, second=0)
    return start, enddt


def analyse_day(day, file):
    inDay = idxs_in_window(day[0], day[1], file['datetime'])
    # afterStart = (file['datetime'] >= day[0]).to_list()
    # beforeEnd = (file['datetime'] <= day[1]).to_list()
    # inDay = [a and b for a,b in zip(afterStart, beforeEnd)]
    temps = file.iloc[inDay]['temperature']

    if len(temps) != 0:
        o30 = [t > 30 for t in temps]
        if np.sum(o30)/len(o30) >= 0.7:
            return inDay
        else:
            return []
    else:
        return []


def split_heatwave(datapath):
    # create new directories if necessary
    heatwavepath = f'{datapath}_heatwave'
    if not os.path.isdir(heatwavepath):
        os.mkdir(heatwavepath)
    nonheatwavepath = f'{datapath}_standard'
    if not os.path.isdir(nonheatwavepath):
        os.mkdir(nonheatwavepath)

    # read first and last day
    for filename in os.listdir(datapath):
        file = pd.read_csv(os.path.join(datapath, filename), delimiter=';')
        file['datetime'] = pd.to_datetime(file['datetime'])
        day, endday = day_interval(file['datetime'].to_list())

        # Begin with all idxs False and gather heatwave measurements
        heatwave = [False] * len(file)
        while day[0] <= endday:
            idxs = analyse_day(day, file)
            if idxs:
                heatwave = [a or b for a, b in zip(heatwave, idxs)]
            day[0] += pd.Timedelta(days=1)
            day[1] += pd.Timedelta(days=1)

        # invert for non-heatwave measurements
        nonheatwave = [not h for h in heatwave]

        # gather non- and heatwave days and save to csv
        if len(file) != len(heatwave):
            x = 5
        heatwavedata = file.iloc[heatwave]
        heatwavedata.to_csv(os.path.join(heatwavepath, filename), index=False)
        nonheatwavedata = file.iloc[nonheatwave]
        nonheatwavedata.to_csv(os.path.join(nonheatwavepath, filename), index=False)


def sensorsplit_prep(loraindir, sensiriondir):
    # save paths for sensor types
    if not os.path.isdir(loraindir):
        os.mkdir(loraindir)
    if not os.path.isdir(sensiriondir):
        os.mkdir(sensiriondir)

    # station information files
    station_info = pd.read_csv('Data/Stations_Zürich_v3.2_LCZ.csv', delimiter=';')
    station_info['Station ID:'] = station_info['Station ID:'].map(normalise_umlauts)
    station_info['Station ID:'] = station_info['Station ID:'].map(remove_operators)
    station_info['Datum und Uhrzeit der Montage:'] = pd.to_datetime(station_info['Datum und Uhrzeit der Montage:'])
    stations = pd.read_csv('Data/stations.csv', sep=';')
    stations['gridcoord'] = stations['gridcoord'].map(normalise_umlauts)
    stations['gridcoord'] = stations['gridcoord'].map(remove_operators)

    return station_info, stations


def copy_file(src, stype):
    loraindir = 'Data/MeasurementFeatures_v6_lorain'
    sensiriondir = 'Data/MeasurementFeatures_v6_sensirion'
    if stype == 'Sensirion':
        shutil.copy(src, sensiriondir)
    elif stype == 'LoRain':
        shutil.copy(src, loraindir)
    else:
        warn(f'Unexpected sensor type ({stype})')
        return ValueError


def list_equal(sensors):
    sensors = list(sensors)
    equal = [x == sensors[0] for x in sensors]
    return np.sum(equal) == len(sensors)


def measurement_split(stationpath, rows, stationid):
    rows.reset_index(drop=True, inplace=True)
    stationdata = pd.read_csv(stationpath, sep=';', parse_dates=['datetime'])
    # empty dfs to put sensor data in
    sensirion = empty_df(stationdata.columns)
    lorain = empty_df(stationdata.columns)

    start = stationdata.loc[0]['datetime']
    end = stationdata.loc[len(stationdata)-1]['datetime']

    # catch measurements that are from before the first recorded Montage for that station
    t0 = rows.iloc[0]['Datum und Uhrzeit der Montage:']
    t1 = rows.iloc[1]['Datum und Uhrzeit der Montage:']
    if start < t0:
        n = np.sum(stationdata['datetime'] < rows.iloc[0]['Datum und Uhrzeit der Montage:'])
        warn(f'{n} measurements do not have a sensor type associated ({stationid})')

    # iterate through all rows, adding measurements to corresponding sensor df
    for idx, row in rows.iterrows():
        if row['Sensor'] == 'Sensirion':
            inWindow = idxs_in_window(t0, t1, stationdata['datetime'])
            sensirion = pd.concat((sensirion, stationdata[inWindow]))
        elif row['Sensor'] == 'LoRain':
            inWindow = idxs_in_window(t0, t1, stationdata['datetime'])
            lorain = pd.concat((lorain, stationdata[inWindow]))

        t0 = row['Datum und Uhrzeit der Montage:']
        if idx == len(rows)-1:
            t1 = end
        else:
            t1 = rows.iloc[idx+1]['Datum und Uhrzeit der Montage:']

    return sensirion, lorain


def split_sensors(datapath):
    loraindir = 'Data/MeasurementFeatures_v6_lorain'
    sensiriondir = 'Data/MeasurementFeatures_v6_sensirion'
    station_info, stations = sensorsplit_prep(loraindir, sensiriondir)

    for filename in os.listdir(datapath):
        stationid = filename.split('.csv')[0]
        try:
            # determine gridcoord ID (used to identify sensor type)
            rowindex = stations.loc[stations['stationid_new'] == stationid].index[0]
            altid = stations['gridcoord'].iloc[rowindex]
        except IndexError:
            warn(f'No gridcoord found for station id {stationid}')
            continue

        # identify row(s) with information on station sensor type
        rowidxs = station_info.loc[station_info['Station ID:'] == altid].index

        if not len(rowidxs):
            warn(f'No sensor information found for station {stationid}')
            continue
        elif len(rowidxs) == 1 or list_equal(station_info['Sensor'].iloc[rowidxs]):
            sensortype = station_info['Sensor'].iloc[rowidxs[0]]
            try:
                copy_file(os.path.join(datapath, filename), sensortype)
            except ValueError:
                continue

        else:
            sensirion, lorain, = measurement_split(os.path.join(datapath, filename), station_info.iloc[rowidxs], stationid)
            sensirion.to_csv(os.path.join(sensiriondir, filename), index=False)
            lorain.to_csv(os.path.join(loraindir, filename), index=False)


def split_dataset(datapath, type):
    if type == 'night':
        split_day_night(datapath)
    elif type == 'heatwave':
        split_heatwave(datapath)
    elif type == 'sensortype':
        split_sensors(datapath)


if __name__ == '__main__':
    datapath = 'Data/MeasurementFeatures_v6'
    type = 'heatwave'
    split_dataset(datapath, type)
