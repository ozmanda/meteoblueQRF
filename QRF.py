import os
from pickle import dump
from sklearn.utils import shuffle
import numpy as np
from pandas import DataFrame, concat
from utils import start_timer, end_timer, mse
from datetime import date
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
        modeldir = os.path.join(os.path.dirname(savedir), f'Trained_Models/{date.today()}_{self.MSE}.pickle')
        savedir = os.path.join(savedir, 'Training')
        if not os.path.isdir(savedir):
            os.mkdir(savedir)
        savedir = os.path.join(savedir, f'{date.today()}_{self.MSE}.csv')

        output = {}
        for featurekey in self.xTest.keys():
            output[featurekey] = self.xTest[featurekey]
        output['Prediction'] = self.yPred
        output['True Temperature'] = self.yTest

        for key in output.keys():
            output[key] = list(output[key])
        output_df = DataFrame(output)

        # save data output and model
        output_df.to_csv(savedir, index=False)
        dump(self.qrf, open(modeldir, "wb"))
        # dump(self.qrf, modeldir)
