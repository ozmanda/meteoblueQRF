import os
import argparse
import numpy as np
import qrf_utils
from qrf_utils import load_csv, sd_mu, load_file
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import read_csv, Timedelta, Timestamp, to_datetime, DataFrame


def extract_inference_times(inferencedatapath):
    file = read_csv(inferencedatapath)
    return list(file['datetime'])


def roundTime(dt, roundTo=5*60):
    """
    Round a datetime object to any timelapse in seconds
    dt : datetime.datetime object, default now.
    roundTo : Closest number of seconds to round to, default 5 minutes.
    Author: Thierry Husson 2012 - Use it as you want but don't blame me.
    """
    dt = Timestamp.to_pydatetime(dt)
    seconds = (dt.replace(tzinfo=None) - dt.min).seconds
    rounding = (seconds+roundTo/2) // roundTo * roundTo
    return to_datetime(dt + Timedelta(seconds=rounding-seconds, microseconds=-dt.microsecond))


def remove_duplicates(featuremap: dict):
    temp = []
    idxs = []
    dupes = []
    for idx, time in enumerate(featuremap['datetime'][:, 0, 0]):
        if time not in temp:
            temp.append(time)
            idxs.append(idx)
        else:
            dupes.append(time)
    print(f'Duplicate times present: \n{dupes}')
    unique = {'datetime': featuremap['datetime'][idxs], 'temperature': featuremap['temperature'][idxs]}
    return unique, idxs


def match_times(times: list, measurementpath: str, stationid: str):
    ''' Returns measurement file rows which correspond to the time inputs. '''
    measurementpath = os.path.join(measurementpath, f'temp_{stationid}.csv')
    try:
        measurementfile = read_csv(measurementpath, delimiter=';')
    except FileNotFoundError:
        # print(f'No measurement file for station {stationid} could be found, skipping')
        return DataFrame({'': []}), None
    try:
        dts = list(measurementfile['datetime'])
    except KeyError:
        measurementfile = read_csv(measurementpath, delimiter=',')
        dts = list(measurementfile['datetime'])

    for idx, time in enumerate(dts):
        dts[idx] = roundTime(to_datetime(time)).tz_localize('UTC')
    measurementfile['datetime'] = dts

    measurementidxs = measurementfile['datetime'].isin(times)
    measurements = measurementfile[measurementfile['datetime'].isin(times)]
    matched = [t in list(measurements['datetime']) for t in times]
    return measurements, matched


def load_data(resultpath: str, featuremappath: str):
    '''
    Loads predicted and true temperature maps
    '''
    featuremap = load_file(featuremappath)
    featuremap, idxs = remove_duplicates(featuremap)
    results = load_file(resultpath)[idxs, :, :]
    temps = featuremap['temperature']
    # extract times and save as a list of Timestamps
    times = list(featuremap['datetime'][:, 0, 0])
    for idx, time in enumerate(times):
        times[idx] = to_datetime(time).tz_localize('UTC')
    return results, temps, times


def isolate_idxs(stations: dict):
    rows = []
    cols = []
    for station in stations.keys():
        rows.append(stations[station]['row'])
        cols.append(stations[station]['col'])
    return rows, cols


def loc_idx(boundary, stations, res=32):
    '''
    Determines the indices of the station on the map using the boundary, map resolution and station coordinates.
    (0, 0) is the top left corner, 0 is at boundary['CH_N'] for station latitude and boundary['CH_W'] for station
    longitude. The index coordinates are therefore:
    (boundary['CH_N'] - station_lat, boundary['CH_W'] + station_lon)
    '''
    for stationid in stations.keys():
        stations[stationid]['row'] = int((boundary['CH_N'] - stations[stationid]['lat'])/res)
        stations[stationid]['col'] = int((stations[stationid]['lon'] - boundary['CH_W'])/res)
    return stations


def stations_loc(boundary, stationdata):
    print('Identifying stations within the boundary')
    print(stationdata)
    stationscsv = read_csv(stationdata, delimiter=";")
    stations = {}
    for _, row in stationscsv.iterrows():
        if boundary['CH_W'] <= int(row["CH_E"]) <= boundary['CH_E'] and boundary['CH_S'] <= int(row["CH_N"]) <= boundary['CH_N']:
            stations[row["stationid_new"]] = {'lat': int(row["CH_N"]), 'lon': int(row["CH_E"]),
                                              'LCZ': row["classification"]}

    stations = loc_idx(boundary, stations)
    return stations


def calculate_station_metrics(tpred, stations, times, measurementpath, savepath):
    '''
    Calculate the errors for stations within the predicted temperature map. Measurements corresponding to the
    prediction times are extracted and for each station, a dictionary containing all relevant metrics is generated.
    '''
    csv_savepath = os.path.join(savepath, 'station_metrics.csv')
    # create empty dict with one entry per station and fill with metrics
    station_metrics = {'stationid': [], 'rmse': [], 'mean': [], 'standard deviation': [], 'LCZ': []}
    for station in stations.keys():
        measurements, matched_times = match_times(times, measurementpath, station)
        if not measurements.empty:
            station_metrics['stationid'].append(station)
            # calculate metrics
            preds = tpred[:, stations[station]['row'], stations[station]['col'], 1][matched_times]
            errors = preds - measurements['temp']
            station_metrics['rmse'].append(round(np.sqrt((1 / len(errors)) * (np.sum(errors ** 2))), 2))
            sd, mu = sd_mu(errors)
            station_metrics['mean'].append(mu)
            station_metrics['standard deviation'].append(sd)
            station_metrics['LCZ'].append(stations[station]['LCZ'])
            # generate error distribution graph
            station_error_distribution(station, errors, savepath)
    df = DataFrame(station_metrics)
    df.to_csv(csv_savepath, sep=';', index=False)
    # histplot of station errors
    fig = sns.histplot(station_metrics['rmse'])
    fig.set_title(f'Station Error Distribution')
    fig.set_xlabel('Prediction Error [°C]')
    plt.savefig(os.path.join(savepath, f'station_error_distribution.png'))
    plt.close()


def station_error_distribution(stationid, errors, savepath):
    savepath = os.path.join(savepath, 'StationDistributions')
    if not os.path.isdir(savepath):
        os.mkdir(savepath)
    fig = sns.histplot(errors)
    fig.set_title(f'Error Distribution for station {stationid}')
    fig.set_xlabel('Prediction Error [°C]')
    plt.savefig(os.path.join(savepath, f'{stationid}.png'))
    plt.close()


def station_error_evaluation(pred, boundary, stationinfo, times, measurementpath, savedir):
    ''' Wrapper for evaluation of measurement station errors '''
    # search for stations within boundary
    stations = stations_loc(boundary, stationinfo)
    calculate_station_metrics(pred, stations, times, measurementpath, savedir)
    return stations


def error_map(pred, true):
    '''
    Creates a map of the prediction errors, calculated as the mean true - predicted temperature value. Therefore,
    positive error values indicate that the prediction is too low, and vice-versa.
    '''
    print('    error maps')
    errormap = true - pred[:, :, :, 1]
    return errormap


def error_heatmap(errormap, stations, path):
    '''
    Generates images of the spatial error distribution for all times. Plots the location of measurement stations
    '''
    rows, cols = isolate_idxs(stations)
    path = os.path.join(path, 'ErrorMaps_new_v2')
    if not os.path.isdir(path):
        os.mkdir(path)
    for time in range(errormap.shape[0]):
        savepath = os.path.join(path, f'errormap_t.{time}.png')
        if not os.path.isfile(savepath):
            ax = sns.heatmap(errormap[time, :, :])
            ax.scatter(rows, cols, marker='*', color='blue')
            plt.show()
            plt.savefig(savepath, bbox_inches='tight')
            plt.close()


def error_hist(errormap, path):
    path = os.path.join(path, 'error_distribution.png')
    if not os.path.isfile(path):
        print('    error hist')
        errors = np.ravel(errormap).astype(list)
        fig = sns.histplot(errors)
        fig.set_title('Prediction Error Distribution')
        fig.set_xlabel('Prediction Error [°C]')
        plt.savefig(path)
        plt.close()


def error_metrics(error_map, path):
    '''
    Using the error map, the MSE of the inference process is calculated. The formula for the MSE is as follows:
    (1/len(true)) * sum((true-pred)**2)
    '''
    path = os.path.join(path, 'error_metrics.txt')
    if not os.path.isfile(path):
        print('    error metrics')
        metrics = {}
        metrics['mse'] = round((1/error_map.size) * (np.sum(error_map**2)), 2)
        metrics['rmse'] = round(np.sqrt(metrics['mse']), 2)
        sd, mean = sd_mu(np.ravel(error_map))
        metrics['mean'] = round(mean, 2)
        metrics['sd'] = round(sd, 2)
        # standard deviation

        lines = [f'Mean Square Error: {metrics["mse"]}\n',
                 f'Residual Mean Square Error: {metrics["rmse"]}\n',
                 f'Mean Error: {metrics["mean"]}\n',
                 f'Standard Deviation: {metrics["sd"]}']
        with open(path, 'w') as file:
            file.writelines(lines)
            file.close()


def error_map_evaluation(pred, true, stations, savepath):
    ''' Wrapper for error evaluation of the predicted temperature maps w.r.t. the moving average feature. '''
    # calculate error maps, calculate metrics and generate heatmaps
    errormaps = error_map(pred, true)
    error_metrics(errormaps, savepath)
    error_hist(errormaps, savepath)
    error_heatmap(errormaps, stations, savepath)


def validation_evaluation(result_path, true_path, boundary, stationinfo, measurementpath):
    '''
    Wrapper function for validation evaluation, which consists generation of error maps using the moving average
    feature and measurement station error analysis.
    '''
    # generate new save folder based on the result path
    savepath = os.path.join(os.path.dirname(result_path), f'{os.path.splitext(os.path.basename(result_path))[0]}')
    if not os.path.isdir(savepath):
        os.mkdir(savepath)

    # load prediction and true maps
    print('Loading temperature maps')
    tempmap_pred, tempmap_true, times = load_data(result_path, true_path)

    # evaluate error map and station errors
    print('Evaluating station errors')
    stations = station_error_evaluation(tempmap_pred, boundary, stationinfo, times, measurementpath, savepath)
    print('Evaluating temperature maps')
    error_map_evaluation(tempmap_pred, tempmap_true, stations, savepath)
