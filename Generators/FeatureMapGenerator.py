import pandas as pd
import numpy as np
import os
from warnings import warn

class FeatureMapGenerator():
    def __init__(self, mode: str, measurementpath: str, stationinfo: str, geopath: str, savepath: str, 
                 boundary: list[float] = [], time: str = '', palmfile: str = '', palmhumi: bool = True, palmtemp: bool = True):
        assert mode in ['inference', 'validation'], 'mode must be either "inference" or "validation"'
        self.stationinfo: pd.DataFrame = pd.read_csv(stationinfo, delimiter=';')
        self.geopath: str = geopath
        self.measurementpath: str = measurementpath

        if mode == 'inference':
            assert boundary, 'boundary must be given for inference'
            
            try:
                times = pd.to_datetime(time, format='%Y/%m/%d_%H:%M')
            except ValueError as e:
                warn('Inference times were entered in an unreadable format. Try again with "YYYY/MM/DD_HH:MM"')
                raise e
            self.inferencemaps(boundary, times)
        
        elif mode == 'validation':
            assert os.path.exists(palmfile), 'Valid PALM simulation file must be given'
            self.validationmaps()

    def inferencemaps(self, boundary, times):
        self.savepath: str = f'INFERENCE/{boundary[0]}-{boundary[1]}_{boundary[2]}_{boundary[3]}-' \
                             f'{times[0].replace("/", "-")}_{times[1].replace("/", "-")}'
        if not os.path.isdir(self.savepath):
            os.mkdirs(self.savepath)

    def validationmaps(self):
        pass

