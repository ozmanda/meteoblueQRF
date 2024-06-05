import pandas as pd
import numpy as np
import os
from warnings import warn
import datautils
import geodata
import irradiation
from typing import List

# GLOBAL VARIABLES
RESOLUTION = 16
PRESSURE = 1013.25
GEOFEATURES = ['altitude', 'buildings', 'buildings_10', 'buildings_30', 'buildings_100', 'buildings_200',
               'buildings_500', 'forests', 'forests_10', 'forests_30', 'forests_100', 'forests_200', 'forests_500',
               'pavedsurfaces', 'pavedsurfaces_10', 'pavedsurfaces_30', 'pavedsurfaces_100', 'pavedsurfaces_200',
               'pavedsurfaces_500', 'surfacewater', 'surfacewater_10', 'surfacewater_30', 'surfacewater_100',
               'surfacewater_200', 'surfacewater_500', 'urbangreen', 'urbangreen_10', 'urbangreen_30', 'urbangreen_100',
               'urbangreen_200', 'urbangreen_500'] #! these aren't right, see the convolutions defined in geodata.py
FAULTYSTATIONS = ['C059A2225266', 'C3FD36A6C1BC','D07769DF208C', 'D883D89E6A24', 'D083B9FD07FB', 'D3FE8EEF188C',
                  'D63DFE9B164B','DF15D23E4B15','E2A0DF1A4941','E437CB2AF225','F033A8C6BB79','F4683D808CFB',
                  'F5C16A4B6340', 'EB90524D4F3E', 'EC032D8260EB', 'FCBBD3B1DB2C']

class FeatureMapGenerator():
    '''
    Generates feature maps for inference or validation purposes. 
    For validation, a PALM simulation file exists and is used to evaluate the model. THe boundary and start/end times are given
    by this PALM file. For inference, the boundary and start/end times are given by the user.
    '''
    def __init__(self, mode: str, measurementpath: str, stationinfo: str, geopath: str, savepath: str):
        self.stationinfo: pd.DataFrame = pd.read_csv(stationinfo, delimiter=';')
        self.geopath: str = geopath
        self.measurementpath: str = measurementpath
        self.mode: str = mode
        self.savepath: str = savepath
        self.features: dict = {}

    def generate(self, boundary: List[float] = [], time: str = '', palmfile: str = '', palmhumi: bool = True, palmtemp: bool = False):       
        if self.mode == 'inference':
            assert boundary, 'boundary must be given for inference'
            self.boundary: List[float] = boundary
            try:
                times = pd.to_datetime(time, format='%Y/%m/%d_%H:%M')
                self.times_aware: List[pd.DatetimeIndex] = list(times)
            except ValueError as e:
                warn('Inference times were entered in an unreadable format. Try again with "YYYY/MM/DD_HH:MM"')
                raise e
            self.inferencemaps()

        elif self.mode == 'validation':
            assert os.path.exists(palmfile), 'Valid PALM simulation file must be given'
            self.palmfile: str = palmfile
            self.palmhumi: bool = palmhumi
            self.validationmaps()


# INFERENCE --------------------------------------------------------------------------------------------------------------------------
    def inferencemaps(self):
        self.savepath: str = f'INFERENCE/{self.boundary[0]}-{self.boundary[1]}_{self.boundary[2]}_{self.boundary[3]}-' \
                             f'{self.times_aware[0].replace("/", "-")}_{self.times_aware[1].replace("/", "-")}'
        if not os.path.isdir(self.savepath):
            os.mkdirs(self.savepath)
        # TODO: inference map generation


# VALIDATION -------------------------------------------------------------------------------------------------------------------------
    def validationmaps(self):
        self.set_paths()
        self.boundary, self.times_aware, t_bool = self.extract_palm_data()
        temps, humis = self.palm_surfacedata()
        self.features['temperature'] = temps[t_bool, :, :]
        self.features['humidity'] = humis[t_bool, :, :]
        self.features['moving_average'] = self.moving_average(t_bool)      
        self.geofeatures()
        self.features.update(self.geomaps)
        self.features['irradiation'] = self.irradiation()[t_bool, :, :]
        datetime_map, time_map = self.datetime_maps(t_bool)
        self.features['datetime'] = datetime_map
        self.features['time'] = time_map
        mappath = os.path.join(self.savepath, f'{self.palmname}_featuremaps.z')
        datautils.dump_file(mappath, self.features) #TODO: check savepath
        print(f'Feature maps generated and saved at {mappath}')


    def set_paths(self):
        self.palmname = os.path.basename(self.palmfile).split(".nc")[0]
        self.folder = os.path.join(self.savepath, f'{self.palmname}_intermediate')
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        self.savefile = f'{os.path.basename(self.palmfile).split(".nc")[0]}_featuremaps.json'


    def datetime_maps(self, t_bool: List[bool]):
        # create time and datetime maps
        self.date_time_sep()
        shape = self.features['temperature'].shape
        datetime_map = np.empty(shape=shape, dtype=np.dtype('U20'))
        time_map = np.empty(shape=shape, dtype=np.dtype('U20'))
        for idx, dt in enumerate(self.times_unaware):
            datetime_map[idx, :, :] = str(dt)
            time_map[idx, :, :] = str(self.times_only[idx])
        return datetime_map[t_bool, :, :], time_map[t_bool, :, :]
    
    
    def date_time_sep(self):
        # separation of time and datetime
        self.times_unaware = []
        self.times_only = []
        for idx, time in enumerate(self.times_aware):
            t = time.tz_localize(None)
            self.times_unaware.append(t)
            self.times_only.append(t.time())
    

    def extract_palm_data(self):
        print('Extracting PALM temperature data..................')
        boundary, times, t_bool = datautils.extract_palm_data(self.palmfile, RESOLUTION)
        datautils.dump_file(f'{os.path.splitext(self.palmfile)[0]}_boundary.z', boundary)
        return boundary, times, t_bool
    

    def palm_surfacedata(self):
        print('PALM surface temperatures.........................')
        if os.path.isfile(os.path.join(self.folder, 'PALM_surfacetemps.z')):
            temps = datautils.load_file(os.path.join(self.folder, 'PALM_surfacetemps.z'))
            humis = datautils.load_file(os.path.join(self.folder, 'PALM_surfacehumis.z'))
        else:
            temps, humis = datautils.extract_surfacetemps(self.palmfile)
            datautils.dump_file(os.path.join(self.folder, 'PALM_surfacetemps.z'), temps)
            datautils.dump_file(os.path.join(self.folder, 'PALM_surfacehumis.z'), humis)
        return temps, humis #TODO: check typing
    

    def stations_loc(self):
        print('Identifying stations within the boundary')
        stationscsv = pd.read_csv(self.stationdata, delimiter=";")
        stations = {}
        for _, row in stationscsv.iterrows():
            if row["stationid_new"] in FAULTYSTATIONS:
                continue
            if self.boundary['CH_W'] <= int(row["CH_E"]) <= self.boundary['CH_E'] and self.boundary['CH_S'] <= int(row["CH_N"]) <= self.boundary['CH_N']:
                stations[row["stationid_new"]] = {'lat': int(row["CH_N"]), 'lon': int(row["CH_E"])}
        return stations #TODO: check typing


    def featuregeneration(self):
        # generate geofeatures for each station and adds them to a dictionary
        features: dict = {}
        features['temperature'] = self.tempgen()


    def ma_temp(self):
        print('Moving Average Temperature........................')
        return datautils.moving_average(self.features['temperature'], self.features['times'])
    

    def moving_average(self, t: List[bool]):
        """
        Performs moving average calculation using the array of surface temperatures using the boolean list 
        indicating for which times a MA exists --> implies the stride length for the moving average calculation,
        keeps it generalised.
        :param surfacetemps: path to PALM simulation file
        :param t: list of boolean values indicating the existance of MA values
        :return: 3-dimensional moving-average temperature map
        """
        # iterate through all times
        stride_ma = np.sum([not x for x in t])
        ma = np.zeros(shape=(len(t), self.features['temperature'].shape[1], self.features['temperature'].shape[2]))
        for time_idx in range(ma.shape[0]-stride_ma):
            ma[time_idx+stride_ma, :, :] = np.mean(self.features['temperature'][time_idx:time_idx+stride_ma, :, :], axis=0)
        return ma


    def geofeatures(self, t_bool: List[bool]):
        print('Generating geofeatures............................')
        if os.path.isfile(os.path.join(self.folder, 'geomaps.z')):
            self.geomaps: dict = datautils.load_file(os.path.join(self.folder, 'geomaps.z'))
        else:
            geomaps_full: dict = geodata.geogen(self.geopath, self.boundary, self.features['humidity'].shape[1], self.features['humidity'].shape[2])
            self.geomaps = {}
            for key in self.geomaps:
                self.geomaps[key] = geomaps_full[key][t_bool, :, :]
            datautils.dump_file(os.path.join(self.folder, 'geomaps.z'), self.geomaps)


    def irradiation(self):
        print('Calculating irradiation...........................')
        if os.path.isfile(os.path.join(self.folder, 'irrad.z')):
            irrad = datautils.load_file(os.path.join(self.folder, 'irrad.z'))
        else:
            irrad = irradiation.irradiationmap(self.boundary, self.times_aware, self.geomaps[0, 0, :, :])
            datautils.dump_file(os.path.join(self.folder, 'irrad.z'), irrad)
            return irrad