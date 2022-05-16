import os
from sklearn.utils import shuffle
import numpy as np
from pandas import DataFrame, concat
from utils import start_timer, end_timer, mse
from quantile_forest import RandomForestQuantileRegressor


class QRF:
    def __init__(self, dataTrain, dataTest):
        # shuffle data
        dataTrain = shuffle(dataTrain)
        dataTest = shuffle(dataTest)

        # assign data
        self.yTrain = dataTrain['temperature']
        self.xTrain = dataTrain.drop(['datetime', 'time', 'temperature'], axis=1)
        self.yTest = dataTest['temperature']
        self.xTest = dataTest.drop(['datetime', 'time', 'temperature'], axis=1)

        # variables used later on
        self.yPred = 0
        self.MSE = 0

    def run_training(self):
        self.qrf = RandomForestQuantileRegressor()
        print('  Training QRF Regressor....     ', end='')
        start_timer()
        self.qrf.fit(self.xTrain, self.yTrain)
        end_timer()

    def run_inference(self):
        print('  Predicting test set....     ', end='')
        start_timer()
        self.yPred = self.qrf.predict(self.xTest, quantiles=[0.025, 0.5, 0.975])
        end_timer()

        self.MSE = mse(self.yTest, self.yPred)


    def save_ouput(self, savedir):
        output = {}
        for featurekey in self.xTest.keys():
            output[featurekey] = self.xTest[featurekey]
        output['Prediction'] = self.yPred
        output['True Temperature'] = self.yTest

        output_df = DataFrame(output)
        output_df.to_csv(savedir, index=False)
