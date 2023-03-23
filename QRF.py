import os
import json
import pickle

import joblib
import warnings
import numpy as np
import sklearn.utils
import _pickle as CPickle
from datetime import datetime
from sklearn.utils import shuffle
from pandas import DataFrame, concat

import utils
from utils import start_timer, end_timer, mse
from sklearn.model_selection import train_test_split
from quantile_forest import RandomForestQuantileRegressor


class QRF:
    def __init__(self):
        # variables used later on
        self.yPred = 0
        self.MSE = 0

    def set_data(self, dataTrain, dataTest):
        # shuffle data
        dataTrain = shuffle(dataTrain)
        dataTest = shuffle(dataTest)

        # assign data
        self.yTrain = dataTrain['temperature']
        self.xTrain = dataTrain.drop(['datetime', 'time', 'temperature'], axis=1)
        self.yTest = dataTest['temperature']
        self.test_times = dataTest['datetime']
        self.xTest = dataTest.drop(['datetime', 'time', 'temperature'], axis=1)

    def set_split_data(self, data):
        self.data = data
        x = self.data.drop(['time', 'temperature'], axis=1)
        y = self.data['temperature']
        self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(x, y, test_size=0.2, random_state=42)

        # extract time and remove as feature variable
        self.test_times = self.xTest['datetime']
        self.xTrain = self.xTrain.drop(['datetime'], axis=1)
        self.xTest = self.xTest.drop(['datetime'], axis=1)

    def run_training(self):
        self.qrf = RandomForestQuantileRegressor()
        print('  Training QRF Regressor....     ', end='')
        start_timer()
        self.qrf.fit(self.xTrain, self.yTrain)
        end_timer()

    def run_test(self):
        print('  Predicting test set....     ', end='')
        start_timer()
        self.yPred = self.qrf.predict(self.xTest, quantiles=[0.025, 0.5, 0.975])
        end_timer()

        self.MSE = mse(self.yTest, self.yPred)

    def run_inference(self, datapath, savedir):
        # open file, load featuremap and close the data file
        data = utils.load_inferencefile(datapath)
        _ = data.pop('datetime')
        _ = data.pop('time')
        if 'moving average' in data.keys():
            data['moving_average'] = data['moving average']
            _ = data.pop('moving average')
        self.xTest, map_shape = utils.unravel_data(data)

        # begin and time
        print('  Predicting inference data....     ', end='')
        start_timer()
        self.yPred = self.qrf.predict(self.xTest, quantiles=[0.025, 0.5, 0.975])
        end_timer()

        prediction_map = utils.reshape_preds(self.yPred, map_shape)
        timenow = datetime.now().replace(second=0, microsecond=0)
        timenow = f'{timenow.year}-{timenow.month}-{timenow.day}_{timenow.hour}.{timenow.minute}'
        savedir = os.path.join(savedir, f'{timenow}.json')

        with open(savedir, 'wb') as file:
            CPickle.dump(prediction_map, file, protocol=pickle.HIGHEST_PROTOCOL)
            file.close()

        return savedir

    def write_variable_importance(self, modelpath, filename):
        print(f'    Writing variable importance file......')
        filepath = os.path.join(modelpath, f'{filename.split(".z")[0]}_variable_importance.txt')
        lines = []
        for key in self.variable_importance.keys():
            lines.append(f'{key}:\t {self.variable_importance[key]}\n')

        with open(filepath, 'w') as file:
            file.writelines(lines)

    def run_variable_importance_estimation(self, modelpath, filename, n=10):
        # ignores UserWarning "X does not have valid feature names, but RandomForestQuantileRegressor
        # was fitted with feature names" and FutureWarning (iteritems -> .items and series[i:j] -> label-based indexing)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            print(f'  Estimating variable importance')
            variables = list(self.xTrain.columns)
            oob_preds = self.qrf.predict(self.xTrain, oob_score=True)
            og_oob_error = np.mean(np.abs(oob_preds - self.yTrain))
            diffsum = 0
            oob_errors = {}
            self.variable_importance = {}
            print(f'    Gathering error differences...........')
            for variable in variables:
                xtrain = self.xTrain
                var_oob_errors = []
                for i in range(n):
                    xtrain[variable] = sklearn.utils.shuffle(self.xTrain[variable]).values
                    var_oob_preds = self.qrf.predict(xtrain, oob_score=True)
                    var_oob_errors.append(np.mean(np.abs(var_oob_preds - self.yTrain)))
                oob_errors[variable] = np.mean(var_oob_errors) - og_oob_error
                diffsum += oob_errors[variable]

            print(f'    Calculating variable importances.......')
            for variable in variables:
                self.variable_importance[variable] = np.round((oob_errors[variable] / diffsum)*100, 2)
                print(f'      {variable}:\t {self.variable_importance[variable]}')

            self.write_variable_importance(modelpath, filename)

    def save_model(self, modelpath):
        timenow = datetime.now().replace(second=0, microsecond=0)
        timenow = f'{timenow.year}-{timenow.month}-{timenow.day}_{timenow.hour}.{timenow.minute}'
        joblib.dump(self, os.path.join(modelpath, f'{timenow}_{self.MSE}.z'),
                    compress=3)

    def save_ouput(self, savedir, modelpath):
        self.save_model(modelpath)
        if not os.path.isdir(savedir):
            os.mkdir(savedir)

        timenow = datetime.now().replace(second=0, microsecond=0)
        timenow = f'{timenow.year}-{timenow.month}-{timenow.day}_{timenow.hour}.{timenow.minute}'
        savedir = os.path.join(savedir, f'{timenow}_{self.MSE}.csv')

        output = {'datetime': self.test_times}
        for featurekey in self.xTest.keys():
            output[featurekey] = self.xTest[featurekey]
        output['Prediction'] = self.yPred
        output['True Temperature'] = self.yTest

        for key in output.keys():
            output[key] = list(output[key])
        output_df = DataFrame(output)

        # save data output and model
        output_df.to_csv(savedir, index=False)
