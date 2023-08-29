import os
import json
import pickle
import time
import joblib
import warnings
import qrf_utils
import numpy as np
import sklearn.utils
import seaborn as sns
from PIL import Image
import _pickle as cPickle
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from pandas import DataFrame, concat
from sklearn.model_selection import train_test_split
from validation_evaluation import validation_evaluation
from quantile_forest import RandomForestQuantileRegressor
from qrf_utils import start_timer, end_timer, mse, load_file, save_object


class QRF:
    def __init__(self, confidence_interval=None):
        # variables used later on
        self.yPred = 0
        self.MSE = 0
        if confidence_interval:
            self.lowerCI = (100 - CI) / 2
            self.upperCI = 100 - self.lowerCI
        self.qrf = RandomForestQuantileRegressor
        self.xTrain = []
        self.xTest = []
        self.yTrain = []
        self.yTest = []
        self.yPred = []
        self.test_times = []
        self.variable_importance = {}
        self.data = DataFrame

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
        self.yPred = self.qrf.predict(self.xTest, quantiles=[self.lowerCI, 0.5, self.upperCI])
        end_timer()

        self.MSE = mse(self.yTest, self.yPred)

    def run_inference(self, datapath, savedir, img=True):
        # open file, load featuremap and close the data file
        self.xTest, map_shape = qrf_utils.load_inference_data(datapath)

        # begin and time
        print('Predicting inference data....     ', end=' ')
        start_timer()
        self.yPred = self.qrf.predict(self.xTest, quantiles=[self.lowerCI, 0.5, self.upperCI])
        end_timer()

        prediction_map = qrf_utils.reshape_preds(self.yPred, map_shape)
        timenow = datetime.now().replace(second=0, microsecond=0)
        timenow = f'{timenow.year}-{timenow.month}-{timenow.day}_{timenow.hour}.{timenow.minute}'
        savedir = os.path.join(savedir, f'{timenow}.json')

        tic = time.perf_counter()
        save_object(savedir, prediction_map)
        toc = time.perf_counter()
        print(f'    save time {toc-tic:0.2f} seconds')

        if img:
            print('Generating images...')
            savedir = os.path.join(os.path.dirname(savedir), timenow, 'prediction_maps')
            qrf_utils.map_vis(prediction_map, savedir)

        return savedir

    def run_validation(self, datapath, measurementpath, palmpath, resultpath=None, run_inference=True):
        """
        Performs a validation run. A validation run consists of loading the feature maps, performing inference and
        evaluating the resulting temperature maps w.r.t. the moving average feature and analysing the accuracy per
        measurement station.
        """
        if run_inference:
            resultpath = self.run_inference(datapath, resultpath)
        else:
            assert resultpath, 'If inference is not run, a path to inference results must be given'
        # load boundary
        boundary = load_file(f'{os.path.splitext(palmpath)[0]}_boundary.z')
        # infopath = 'Data/stations.csv'
        infopath = os.path.join(os.path.dirname(os.path.dirname(measurementpath)), 'stations.csv')
        validation_evaluation(resultpath, datapath, boundary, infopath, measurementpath)


    def generate_images(self, inferencefile, imgpath):
        """
        Generates images suitable for loading into SR_GAN for training or testing. Note that the inference file loaded
        contains three prediction values: [CI lower bound, mean prediction, CI upper bound]. Currently only the mean
        prediction value is used to generate images.

        Normalisation: normalised using (nan)max/min temp values and adding 5° buffer on each end.
        """
        inference_maps = load_file(inferencefile)
        vmin = np.nanmin(inference_maps) - 5
        vmax = np.nanmax(inference_maps) + 5

        for time in range(inference_maps.shape[0]):
            tempmap = sns.heatmap(inference_maps[time, :, :, 1], vmin=vmin, vmax=vmax, linewidth=0, cbar=False,
                                  yticklabels=False, xticklabels=False)
            plt.close()
            fig = tempmap.get_figure()
            fig.savefig(os.path.join(imgpath, f'tempmap_{time}.png'), bbox_inches='tight', pad_inches=0)

    def write_variable_importance(self, filepath):
        print(f'    Writing variable importance file......')
        filepath = f'{filepath.split(".z")[0]}_variable_importance.txt'
        lines = []
        for key in self.variable_importance.keys():
            lines.append(f'{key}:\t {self.variable_importance[key]}\n')

        with open(filepath, 'w') as file:
            file.writelines(lines)

    def run_variable_importance_estimation(self, filepath, n=10):
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
                for _ in range(n):
                    xtrain[variable] = sklearn.utils.shuffle(self.xTrain[variable]).values
                    var_oob_preds = self.qrf.predict(xtrain, oob_score=True)
                    var_oob_errors.append(np.mean(np.abs(var_oob_preds - self.yTrain)))
                oob_errors[variable] = np.mean(var_oob_errors) - og_oob_error
                diffsum += oob_errors[variable]

            print(f'    Calculating variable importances.......')
            for variable in variables:
                self.variable_importance[variable] = np.round((oob_errors[variable] / diffsum)*100, 2)
                print(f'      {variable}:\t {self.variable_importance[variable]}')

            self.write_variable_importance(filepath)

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
