import numpy as np
import pandas as pd
from warnings import warn
import time
import pickle
from metpy.calc import relative_humidity_from_mixing_ratio
from metpy.units import units
from typing import List
import netCDF4 as nc

PRESSURE = 1013.25

# COORDINATES -----------------------------------------------------------------
def lv95_to_lv03(lv95_lat: float, lv95_lon: float):
    return lv95_lat - 1000000, lv95_lon - 2000000

def lv03_to_lv95(lv03_lat: float, lv03_lon: float):
    return lv03_lat + 1000000, lv03_lon + 2000000

def lv_to_wgs84(lv_lat: float, lv_lon: float, type: float, h_lv: float = 0):
    if type == 'lv03':
        y_prime: float = (lv_lon - 600000) / 1000000
        x_prime: float = (lv_lat - 200000) / 1000000
    elif type == 'lv95':
        y_prime: float = (lv_lon - 2600000) / 1000000
        x_prime: float = (lv_lat - 1200000) / 1000000
    else:
        warn(f'Invalid type ({type}) passed for conversion (only "lv95" or "lv03" accepted).')
        raise ValueError

    lambda_prime: float = 2.6779094 + \
                   4.728982 * y_prime + \
                   0.791484 * y_prime * x_prime + \
                   0.130600 * y_prime * x_prime**2 - \
                   0.043600 * y_prime**3

    phi_prime: float = 16.9023892 + \
                3.238272 * x_prime - \
                0.270978 * y_prime**2 - \
                0.002528 * x_prime**2 - \
                0.044700 * x_prime + y_prime**2 - \
                0.014000 * x_prime**3

    wgs84_lat: float = (phi_prime * 100) / 36
    wgs84_lon: float = (lambda_prime * 100) / 36

    if h_lv:
        h_wgs: float = h_lv + 49.55 \
                - 12.9 * y_prime \
                - 22.64 * x_prime
        return wgs84_lat, wgs84_lon, h_wgs
    else:
        return wgs84_lat, wgs84_lon


def wgs84_to_lv(wgs84_lat: float, wgs84_lon: float, type: str, 
                h_wgs: float = 0, unit: str = 'deg'):
    if unit == 'deg':
        wgs84_lat *= 3600
        wgs84_lon *= 3600
    # Breite = latitude = phi, Länge = longitude = lambda
    phi_prime = (wgs84_lat - 169028.66) / 10000
    lambda_prime = (wgs84_lon - 26782.5) / 10000


    # E = longitude, N = latitude
    lv95_lon: float =  2600072.37 \
                + 211455.93 * lambda_prime \
                - 10938.51 * lambda_prime * phi_prime \
                - 0.36 * lambda_prime * phi_prime**2 \
                - 44.54 * lambda_prime**3

    lv95_lat: float = 1200147.07 \
               + 308807.95 * phi_prime \
               + 3745.25 * lambda_prime**2 \
               + 76.63 * phi_prime**2 \
               - 194.56 * lambda_prime**2 * phi_prime \
               + 119.79 * phi_prime**3

    if h_wgs:
        h_lv = h_wgs - 49.55 \
               + 2.73 * lambda_prime \
               + 6.94 * phi_prime

    if type == 'lv95' and h_wgs:
        return lv95_lat, lv95_lon, h_lv #type: ignore
    elif type == 'lv95' and not h_wgs:
        return lv95_lat, lv95_lon, 0
    elif type == 'lv03':
        lv03_lat, lv03_lon = lv95_to_lv03(lv95_lat, lv95_lon)
        if h_wgs:
            return lv03_lat, lv03_lon, h_lv #type: ignore
        else:
            return lv03_lat, lv03_lon, 0
        

# TIME -----------------------------------------------------------------------
def roundTime(dt, roundTo=5*60):
    """
    Round a datetime object to any time lapse in seconds
    dt : datetime.datetime object, default now.
    roundTo : Closest number of seconds to round to, default 5 minutes.
    Author: Thierry Husson 2012 - Use it as you want but don't blame me.
    """
    dt = pd.Timestamp.to_pydatetime(dt)
    seconds = (dt.replace(tzinfo=None) - dt.min).seconds
    rounding = (seconds+roundTo/2) // roundTo * roundTo
    return pd.to_datetime(dt + pd.Timedelta(seconds=rounding-seconds, microseconds=-dt.microsecond))


def DST_TZ(times: list):
    for idx, time in enumerate(times):
        if pd.Timestamp('2019-03-31 02:00') <= time <= pd.Timestamp('2019-10-27 02:00'):
            time -= pd.Timedelta(hours=2)
            time = time.tz_localize('utc')
            time = time.tz_convert('Europe/Zurich')
            times[idx] = time
        else:
            time -= pd.Timedelta(hours=1)
            time = time.tz_localize('utc')
            time = time.tz_convert('Europe/Zurich')
            times[idx] = time
    return times


def extract_times(origintime: np.datetime64, times_list: list):
    """
    Extracts the time vector, formatting it as a datetime. The time contained within the PALM file is given as
    minutes since origin. Additionally, a boolean vector is generated, indicating the start of the useable time
    series (certain observations are required to create the moving average).
    """
    times = []
    print(f'times list length: {len(times_list)}')
    for _, time in enumerate(times_list):
        times.append(origintime + pd.Timedelta(minutes=np.round(time * 24 * 60)))

    #* lost observations fixed, for most PALM files it is 2/56 which are lost
    # lost observations for the one hour moving average: 60 min / timedelta
    td_minutes = (times[2]-times[1]).total_seconds() / 60
    lost_obs = int(60/td_minutes)
    t_bool = [True] * len(times)
    t_bool[0:lost_obs] = [False] * lost_obs
    if not times:
        raise ValueError
    return times, t_bool


def start_timer():
    global _start_time
    _start_time = time.time()


def end_timer():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    print(f'Time: {t_hour}:{t_min}:{t_sec}')



# FILE HANDLING --------------------------------------------------------------
def dump_file(path: str, object):
    with open(path, 'wb') as file:
        pickle.dump(object, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.close()


def load_file(path: str):
    with open(path, 'rb') as file:
        object = pickle.load(file)
        file.close()
    return object

def remove_emptytimes(maps: np.ndarray, times: np.ndarray):
    print(f'remove_emptylines before: {times.shape}')
    emptytimes = []
    for time in range(maps.shape[0]):
        if not np.sum(maps[time, :, :]):
            emptytimes.append(time)
            continue
    if emptytimes:
        maps = np.delete(maps, emptytimes, axis=0)
        times = np.delete(times, emptytimes, axis=0)
    print(f'remove_emptylines after: {times.shape}')
    return maps, times


def preprocessing(filepath: str):
    """
    Loads measurements files and executes two preprocessing steps: rounding
    times to 5 minutes and removes duplicate lines.
    """
    csvfile = pd.read_csv(filepath, delimiter=";")
    csvfile['datetime'] = pd.to_datetime(csvfile['datetime'])
    for idx, row in csvfile.iterrows():
        csvfile.iloc[idx, 0] = roundTime(row['datetime'])
    csvfile = csvfile.drop_duplicates()
    return csvfile

def file_matching(tempfile: pd.DataFrame, humifile: pd.DataFrame):
    # new arrays
    newhumi = np.empty(shape=(0, 2))
    newtemp = np.empty(shape=(0, 2))

    if len(tempfile) < len(humifile):
        humitimes = humifile['datetime'].to_list()
        for _, row in tempfile.iterrows():
            try:
                idx = humitimes.index(row['datetime'])
                humidat = [[humitimes[idx], humifile.iloc[idx, 1]]]
                tempdat = [[row['datetime'], row['temp']]]
            except ValueError:
                continue
            newtemp = np.append(newtemp, tempdat, axis=0)
            newhumi = np.append(newhumi, humidat, axis=0)

    elif len(humifile) < len(tempfile):
        temptimes = tempfile['datetime'].to_list()
        for _, row in humifile.iterrows():
            try:
                idx = temptimes.index(row['datetime'])
                tempdat = [[temptimes[idx], tempfile.iloc[idx, 1]]]
                humidat = [[row['datetime'], row['humi']]]
            except ValueError:
                continue
            newtemp = np.append(newtemp, tempdat, axis=0)
            newhumi = np.append(newhumi, humidat, axis=0)

    else:
        warn("file_matching function called unnecessarily", Warning)
        return tempfile, humifile

    newtemp = pd.DataFrame(newtemp, columns=['datetime', 'temp'])
    newhumi = pd.DataFrame(newhumi, columns=['datetime', 'humi'])
    return newtemp, newhumi


def reduce_resolution(original_array: np.ndarray, resolution: int):
    """
    Reduces the dimension of a given array by the resolution. A 10x10 array with a resolution of 2 would return a 5x5 
    array. The method uses a simple average 
    """
    new_array = np.zeros(shape=(int(original_array.shape[0]/resolution), int(original_array.shape[1]/resolution)))
    for row in range(new_array.shape[0]):
        for col in range(new_array.shape[1]):
            arr = original_array[row*resolution:row*resolution+resolution, col*resolution:col*resolution+resolution]
            if np.isnan(arr).all():
                new_array[row, col] = np.nan
            else:
                new_array[row, col] = np.nanmean(arr)
        
    return new_array


def extract_surfacedata(palmpath: str):
    palmfile = pd.Dataset(palmpath, 'r', format='NETCDF4')
    try:
        temps = palmfile['theta_xy']
    except IndexError:
        temps = palmfile['theta']
    all_mr = palmfile['q_xy']
    palmfile.close()

    surf_temps = np.zeros(shape=(temps.shape[0], temps.shape[2], temps.shape[3]))
    surf_humis = np.zeros(shape=surf_temps.shape)

    for time in range(temps.shape[0]):
        for idxs, _ in np.ndenumerate(temps[time, 0, :, :]):
            for layer in range(temps.shape[1]):
                if temps[time, layer, idxs[0], idxs[1]] != -9999:
                    surf_temps[time, :, :][idxs] = temps[time, layer, idxs[0], idxs[1]] - 273.15
                    surface_mixing_ratio = all_mr[time, layer, idxs[0], idxs[1]]
                    temp = palmfile['theta_xy'][time, layer, idxs[0], idxs[1]]
                    relative_humidity = relative_humidity_from_mixing_ratio(PRESSURE*units.hPa,
                                                                            (temp - 273.15) * units.degC,
                                                                            surface_mixing_ratio).to('percent')
                    relative_humidity = round(float(relative_humidity), 2)
                    surf_humis[time, idxs[0], idxs[1]] = relative_humidity
                    break
                else:
                    continue
    # flip maps to account for PALM having origin at the bottom left, not top left
    surf_temps = np.flip(surf_temps, axis=1)
    surf_humis = np.flip(surf_humis, axis=1)

    return surf_temps, surf_humis


def moving_average(temps: List[float], datetimes: list, timedelta=pd.Timedelta(minutes=60)):
    #! this moving average calculate is not correct, it doesn't consider that observations are lost
    movingaverage = []
    for i, time in enumerate(datetimes):
        ma = []
        for idx, t in enumerate(datetimes):
            if time-timedelta <= t <= time:
                ma.append(temps[idx])
            elif t > time:
                break
        if not ma:
            movingaverage.append(temps[i])
        else:
            movingaverage.append(np.mean(ma, axis=0))

    if len(movingaverage) != len(temps):
        warn(f'Shape of moving average vector ({movingaverage.shape}) is not equivalent to the length of the '
             f'temperature vector ({temps.shape})')
        raise ValueError
    return movingaverage


def extract_palm_data(palmpath: str, res: int):
    #* has been checked, times are correct
    """
    Extracts times, temperature and boundary coordinates from PALM file. PALM coordinates are extracted as latitude
    and longitude (WGS84) and converted to LV95 projection coordinates.
    PALM: origin_x contains the longitude, origin_y contains the latitude.
    
    times: array of times
    t: list of boolean values, indicating if a moving-average value is available
    """
    print('Extracting PALM File data....................')
    print('    loading PALM file........................')
    palmfile: nc.Dataset = nc.Dataset(palmpath, 'r', format='NETCDF4')

    print('    determining boundary.....................')
    CH_S, CH_W = lv03_to_lv95(palmfile.origin_y, palmfile.origin_x)
    # CH_S, CH_W, _ = wgs84_to_lv(palmfile.origin_lat, palmfile.origin_lon, 'lv95') #type: ignore
    CH_N = CH_S + palmfile.dimensions['y'].size * res
    CH_E = CH_W + palmfile.dimensions['x'].size * res
    boundary = {'CH_S': CH_S, 'CH_N': CH_N, 'CH_E': CH_E, 'CH_W': CH_W}

    print('    extracting times.........................')
    times, t_bool = extract_times(pd.to_datetime(palmfile.origin_time), palmfile['time'])
    times = np.array(times)

    return boundary, times, t_bool


def extract_surfacetemps(palmpath):
    palmfile = nc.Dataset(palmpath, 'r', format='NETCDF4')
    try:
        temps = palmfile['theta_xy']
    except IndexError:
        temps = palmfile['theta']
    surf_temps = np.zeros(shape=(temps.shape[0], temps.shape[2], temps.shape[3]))
    for time in range(temps.shape[0]):
        for idxs, _ in np.ndenumerate(temps[time, 0, :, :]):
            for layer in range(temps.shape[1]):
                if temps[time, layer, idxs[0], idxs[1]] != -9999:
                    surf_temps[time, :, :][idxs] = temps[time, layer, idxs[0], idxs[1]] - 273.15
                    break
                else:
                    continue
    # flip maps to account for PALM having origin at the bottom left, not top left
    surf_temps = np.flip(surf_temps, axis=1)

    return surf_temps
