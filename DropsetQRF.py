from pandas import DataFrame, concat
from quantile_forest import RandomForestQuantileRegressor

class DropsetQRF():
    def __init__(self, datasets):
        self.data = datasets
        self.stations = datasets.keys()

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
        MSE = {}
        insideCI = {}
        for key in self.stations:
            xTrain, xTest, yTrain, yTest = self.xy_generation(key)
            qrf = RandomForestQuantileRegressor()
            qrf.fit(xTrain, yTrain)

            yPred = qrf.predict(xTest, quantiles=[0.025, 0.5, 0.975])
            MSE['key'] =

