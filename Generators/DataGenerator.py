import numpy as np
import pandas as pd
import os
import datautils
from warnings import warn
from typing import Tuple, List
import geodata
import irradiation as solar

CONVOLUTIONS = [32, 48, 112, 208, 512]
GEOFEATURES = ['altitude', 'buildings', 'forests', 'pavedsurfaces', 'surfacewater', 'urbangreen']
FAULTYSTATIONS = ['C059A2225266', 'C3FD36A6C1BC','D07769DF208C', 'D883D89E6A24', 'D083B9FD07FB', 'D3FE8EEF188C',
                  'D63DFE9B164B','DF15D23E4B15','E2A0DF1A4941','E437CB2AF225','F033A8C6BB79','F4683D808CFB',
                  'F5C16A4B6340', 'EB90524D4F3E', 'EC032D8260EB', 'FCBBD3B1DB2C']


class DataGenerator:
    def __init__(self, datapath: str, geopath: str, savepath: str, infofile: str, convolutions: List[int] = None):
        self.datapath: str = datapath
        self.geodatapath: str = geopath
        self.savepath: str = savepath
        self.infofile: pd.DataFrame = pd.read_csv(infofile, delimiter=';')

    def generate(self):
        for filename in os.listdir(self.datapath):
            if filename.startswith('temp'):
                stationid = filename.split('.csv')[0].split('_')[1]
                savepath = os.path.join(self.savepath, f'{stationid}.csv')
                if os.path.exists(savepath) or stationid in FAULTYSTATIONS:
                    continue
                try:
                    station_df = self.station_dataset(stationid)
                except ValueError: 
                    warn(f'Error in processing station {stationid}, skipping this station', Warning)
                    continue
                station_df.to_csv(savepath, index=False, sep=';')
                print('Done')
                
                    
    def station_dataset(self, stationid: str) -> pd.DataFrame:          
        print(f'Processing station {stationid}........')
        humifile, tempfile = self.loadfiles(stationid)
        humi = humifile['humi'].to_list()
        times = datautils.DST_TZ(tempfile['datetime'].to_list())
        temps = tempfile.reset_index()['temp'].to_list()
        geofeatures, targetlat, targetlon = self.get_geofeatures(stationid, len(temps)) #* keys are correct here
        irradiation = solar.irradiationcalc(times, targetlat, targetlon)
        times, datetimes = self.time_formatting(times)
        ma_temps = datautils.moving_average(temps, datetimes)
        return self.generate_df(datetimes, times, geofeatures, humi, irradiation, temps, ma_temps)    


    def generate_df(self, datetimes, times, geofeatures, humis, irradiation, temps, moving_average) -> pd.DataFrame:
        df = {}
        df['datetime'] = datetimes
        df['time'] = times
        for geofeature in geofeatures.keys():
            df[geofeature] = geofeatures[geofeature]
        df['humidity'] = humis
        df['irradiation'] = irradiation
        df['moving_average'] = moving_average
        df['temperature'] = temps
        df = pd.DataFrame(df)
        return df


    def get_geofeatures(self, stationid, num):
        #! Geofeatures changed from array to dictionary, ensure this works!
        geofeatures = {}
        targetlat, targetlon = geodata.get_loc(stationid, self.infofile)

        for geofeature in GEOFEATURES:
            geofeatures.update(geodata.extract_feature(targetlat, targetlon, geofeature, self.geodatapath, num))

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
            temp, humi = datautils.file_matching(temp, humi)
        
        return humi, temp
    

    def time_formatting(self, times):
        datetime = times.copy()
        for idx, time in enumerate(times):
            times[idx] = time.tz_localize(None).time()
            datetime[idx] = datetime[idx].tz_localize(None)
        return times, datetime