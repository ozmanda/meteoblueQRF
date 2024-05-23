import numpy as np
import pandas as pd
import os
import datautils
from warnings import warn
from typing import Tuple

CONVOLUTIONS = [10, 30, 100, 200, 500]
GEOFEATURES = ['altitude', 'buildings', 'forests', 'pavedsurfaces', 'surfacewater', 'urbangreen']
FAULTYSTATIONS = ['C059A2225266', 'C3FD36A6C1BC','D07769DF208C', 'D883D89E6A24', 'D083B9FD07FB', 'D3FE8EEF188C',
                  'D63DFE9B164B','DF15D23E4B15','E2A0DF1A4941','E437CB2AF225','F033A8C6BB79','F4683D808CFB',
                  'F5C16A4B6340', 'EB90524D4F3E', 'EC032D8260EB', 'FCBBD3B1DB2C']


class DataGenerator:
    def __init__(self, datapath: str, geopath: str, savepath: str, infofile: str, convolutions: list[int] = None):
        self.datapath: str = datapath
        self.geodatapath: str = geopath
        self.savepath: str = savepath
        self.infofile: pd.DataFrame = pd.read_csv(infofile, delimiter=';')
        self.convolutions: list[int] = CONVOLUTIONS

    def generate(self):
        for filename in os.listdir(self.datapath):
            if filename.startswith('temp'):
                stationid = filename.split('.csv')[0].split('_')[1]
                if stationid in FAULTYSTATIONS:
                    continue
                station_df = self.station_dataset(stationid)
                station_df.to_csv(os.path.join(self.savepath, f'{stationid}.csv'), index=False, sep=';')
                print('Done')
                
                    
    def station_dataset(self, stationid: str) -> pd.DataFrame:          
        print(f'Processing station {stationid}........')
        humifile, tempfile = self.loadfiles(stationid)
        humi = humifile['humi'].to_list()
        times = datautils.DST_TZ(tempfile['datetime'].to_list())
        temps = tempfile.reset_index()['temp'].to_list()
        geofeatures, targetlat, targetlon = geofeatures(stationid, len(temps))
        irradiation = irradiation.irradiationcalc(times, targetlat, targetlon)
        times, datetimes = self.time_formatting(times)
        ma_temps = datautils.moving_average(temps, datetimes)
        return self.generate_df(datetimes, times, geofeatures, humi, irradiation, temps, ma_temps)    


    def generate_df(self, datetimes, times, geofeatures, humis, irradiation, temps, moving_average) -> pd.DataFrame:
        df = self.empty_df()
        df['datetime'] = datetimes
        df['time'] = times
        for geofeature in self.geofeatures:
            df[geofeature] = geofeatures[geofeature]
        df['humidity'] = humis
        df['irradiation'] = irradiation
        df['moving_average'] = moving_average
        df['temperature'] = temps
        return df


    def get_geofeatures(self, stationid, num):
        #! Geofeatures changed from array to dictionary, ensure this works!
        geofeatures = {}
        targetlat, targetlon = datautils.get_loc(stationid, self.infofile)

        for geofeature in GEOFEATURES:
            geofeatures[geofeature] = datautils.extract_feature(targetlat, targetlon, geofeature, self.geopath, None, num)

        return geofeatures, targetlat, targetlon

    
    def loadfiles(self, stationid: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            humi: pd.DataFrame = datautils.preprocessing(os.path.join(self.datapath, f'humi_{stationid}.csv'))
        except FileNotFoundError as e:
            warn(f'No humidity file found for {stationid}, skipping this station')
            raise e
        
        try:
            temp: pd.DataFrame = datautils.preprocessing(os.path.join(self.datapath, f'temp_{stationid}.csv'))
        except FileNotFoundError as e:
            warn(f'No temperature file found for {stationid}, skipping this station')
            raise e
        
        if len(humi) != len(temp):
            warn(f'the number of measured temperatures and measured humidities do not coincide, matching...', Warning)
            tempfile, humifile = datautils.file_matching(tempfile, humifile)
        
        return humi, temp


    def empty_df(self) -> pd.DataFrame:
        cols: list[str]  = ['datetime', 'time']
        cols.extend(self.conv_features())
        cols.extend(['humidity', 'irradiation'])
        cols.append('moving_average')
        cols.append('temperature')
        df: pd.DataFrame = pd.DataFrame(columns=cols)
        return df
    

    def conv_features(self) -> list[str]:
        self.geofeatures = []
        for geofeatures in GEOFEATURES:
            for convolution in self.convolutions:
                self.geofeatures.append(f'{geofeatures}_{convolution}')
        return self.geofeatures
    

    def time_formatting(self, times):
        datetime = times.copy()
        for idx, time in enumerate(times):
            times[idx] = time.tz_localize(None).time()
            datetime[idx] = datetime[idx].tz_localize(None)
        return times, datetime