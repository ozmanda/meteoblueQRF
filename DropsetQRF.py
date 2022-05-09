import os
import numpy as np
from pandas import DataFrame, concat
from quantile_forest import RandomForestQuantileRegressor

class DropsetQRF():
    def __init__(self, datasets, CI):
        self.data = datasets
        self.stations = datasets.keys()
        self.lowerCI = (100-CI) / 2
        self.upperCI = 100 - self.lowerCI

    def xy_generation(self, testkey):
        # separate test from training data
        xyTest = self.data[testkey]
        xyTrain = DataFrame(columns=xyTest.columns)
        for key in self.data:
            if key != testkey:
                xyTrain = concat([xyTrain, self.data[key]])

        # test data
        yTest = xyTest['temperature']
        xTest = xyTest.drop(["datetime", 'time', "temperature"], axis=1)
        del xyTest

        # training data
        yTrain = xyTrain['temperature']
        xTrain = xyTrain.drop(["datetime", 'time', "temperature"], axis=1)

        return xTrain.dropna(), xTest, yTrain, yTest

    def run_error_estimation(self):
        self.Output = {}
        for key in self.stations:
            stationData = {}
            xTrain, xTest, yTrain, yTest = self.xy_generation(key)
            qrf = RandomForestQuantileRegressor()
            qrf.fit(xTrain, yTrain)

            # predict test set
            yPred = qrf.predict(xTest, quantiles=[0.025, 0.5, 0.975])

            # calculate relevant error metrics and fill into dictionary
            stationData['True Temperature'] = yTest
            stationData['Predicted Temperature'] = yPred[:, 1]
            stationData['2.5%'] = yPred[:, 0]
            stationData['97.5%'] = yPred[:, 2]
            stationData['Absolute Deviation'] = np.abs(yTest - yPred[:, 1])
            stationData['MSE'] = np.sum(stationData['Absolute Deviation'] ** 2) / len(stationData['Absolute Deviation'])

            # enter station dictionary into ouput dictionary
            self.Output[key] = stationData

    def save_output(self, savepath):
        if not os.path.isdir(savepath):
            os.mkdir(savepath)
        for key in self.Output:
            MSE = self.Output[key].pop('MSE')
            df = DataFrame(self.Output[key])
            df.to_csv(os.path.join(savepath, f'{key}_MSE_{np.round(MSE, 2)}.csv'))
