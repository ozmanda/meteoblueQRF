import os
import numpy as np
from warnings import warn
from pandas import DataFrame, concat
from qrf_utils import start_timer, end_timer, sd_mu
from quantile_forest import RandomForestQuantileRegressor
import joblib


class DropsetQRF:
    def __init__(self, datasets, confidence_interval=None):
        self.data = datasets
        self.stations = datasets.keys()
        CI = confidence_interval if confidence_interval else 95
        self.lowerCI = (100-CI) / 2
        self.upperCI = 100 - self.lowerCI

    def xy_generation(self, testkey):
        # assert the key has data
        assert len(self.data[testkey]) != 0, f'No data available for key {testkey}'
        # separate test from training data
        xyTest = self.data[testkey]
        xyTrain = DataFrame(columns=xyTest.columns)
        for key in self.data:
            if key != testkey:
                xyTrain = concat([xyTrain, self.data[key]])

        # test data
        yTest = xyTest['temperature']
        xTime = xyTest['datetime']
        xTest = xyTest.drop(["datetime", 'time', "temperature"], axis=1)
        del xyTest

        # training data
        yTrain = xyTrain['temperature']
        xTrain = xyTrain.drop(["datetime", 'time', "temperature"], axis=1)

        return xTrain.dropna(), xTest, yTrain, yTest, xTime

    def run_error_estimation(self, savepath, savemodels=True):
        self.Output = {}
        for key in self.stations:
            print(f'Key {key} \n  Generating dataset')
            stationData = {}
            try:
                xTrain, xTest, yTrain, yTest, xTime = self.xy_generation(key)
            except AssertionError:
                warn(f'Skipping station {key}')
                continue
            qrf = RandomForestQuantileRegressor()
            print('  Training QRF Regressor....     ', end='')
            start_timer()
            qrf.fit(xTrain, yTrain)
            end_timer()
            if savemodels:
                joblib.dump(qrf, os.path.join(savepath, f'{key}.z'), compress=3)

            # predict test set
            print('  Predicting test set....     ', end='')
            start_timer()
            yPred = qrf.predict(xTest, quantiles=[self.lowerCI, 0.5, self.upperCI])
            end_timer()

            # calculate relevant error metrics and fill into dictionary
            print('  Generating output dataset....')
            stationData['datetime'] = xTime
            stationData['True Temperature'] = yTest
            stationData['Predicted Temperature'] = yPred[:, 1]
            stationData['2.5%'] = yPred[:, 0]
            stationData['97.5%'] = yPred[:, 2]
            stationData['Deviation'] = yTest - yPred[:, 1]
            stationData['Absolute Deviation'] = np.abs(stationData['Deviation'])
            stationData['MSE'] = np.sum(stationData['Absolute Deviation'] ** 2) / len(stationData['Absolute Deviation'])
            sd, mu = sd_mu(stationData['Deviation'])
            stationData['SD'] = sd
            stationData['ME'] = mu

            # enter station dictionary into output dictionary
            self.Output[key] = stationData


    def save_output(self, savepath):
        station_summary = {'Station': [], 'MSE': [], 'Standard Deviation': [], 'Mean Error': []}
        if not os.path.isdir(savepath):
            os.mkdir(savepath)
        for key in self.Output:
            station_summary['Station'].append(key)
            station_summary['MSE'].append(self.Output[key].pop('MSE'))
            station_summary['Standard Deviation'].append(self.Output[key].pop('SD'))
            station_summary['Mean Error'].append(self.Output[key].pop('ME'))
            DataFrame(self.Output[key]).to_csv(os.path.join(savepath, 'errors', f'errors_{key}.csv'), index=False)

        DataFrame(station_summary).to_csv(os.path.join(savepath, 'station_summary.csv'), index = False)

    def run_dropset_estimation(self, savepath, savemodels=True):
        self.run_error_estimation(os.path.join(savepath, 'models'), savemodels=savemodels)
        self.save_output(savepath)
