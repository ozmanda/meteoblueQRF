import os
import json
import pickle
import time
import joblib
import warnings
import numpy as np
import sklearn.utils
import seaborn as sns
from PIL import Image
import pickle as cPickle
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from pandas import DataFrame, concat
from sklearn.model_selection import train_test_split
from validation_evaluation import validation_evaluation
from quantile_forest import RandomForestQuantileRegressor
from qrf_utils import *

non_training_variables = ['datetime', 'time', 'temperature', 'stationid']


class QRF:
    def __init__(self, modelname=None, confidence_interval=None):
        # variables used later on
        if modelname:
            self.modelname = modelname
        self.yPred = 0
        self.MSE = 0
        CI = confidence_interval if confidence_interval else 95
        self.lowerCI = ((100 - CI) / 2)/100
        self.upperCI = (100 - self.lowerCI)/100
        self.qrf = RandomForestQuantileRegressor
        self.xTrain = []
        self.xTest = []
        self.yTrain = []
        self.yTest = []
        self.yPred = []
        self.test_times = []
        self.variable_importance = {'Variable': [], 'Importance [%]': []}
        self.data = DataFrame


    def load_training_data(self, path, start=None, end=None, test_split=False):
        dataset = load_data(path, startDatetime=start, endDatetime=end)
        if test_split:
            self.set_split_data(dataset)
        else:
            self.set_training_data(dataset)
    

    def load_test_data(self, path, start=None, end=None):
        dataset = load_data(path, startDatetime=start, endDatetime=end)
        self.set_test_data(dataset)


    def set_training_data (self, dataTrain):
        assert len(dataTrain) != 0, 'Cannot set en empty training set'
        dataTrain = shuffle(dataTrain)
        self.yTrain = dataTrain['temperature']
        self.xTrain = dataTrain.drop(non_training_variables, axis=1)
        self.train_stations = dataTrain['stationid']
        self.train_times = dataTrain['datetime']


    def set_test_data(self,  dataTest):
        assert len(dataTest) != 0, 'Cannot set en empty test set'
        dataTest = shuffle(dataTest)
        self.yTest = dataTest['temperature']
        self.xTest = dataTest.drop(non_training_variables, axis=1)
        self.test_times = dataTest['datetime']
        self.test_stations = dataTest['stationid']


    def set_split_data(self, dataset):
        self.data = dataset
        x = self.data.drop(['time', 'temperature'], axis=1)
        y = self.data['temperature']
        self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(x, y, test_size=0.2, random_state=42)

        # extract time and station ID and remove as feature variable (not saved for training data)
        self.test_stations = self.xTest['stationid']
        self.test_times = self.xTest['datetime']
        self.xTrain = self.xTrain.drop(['datetime', 'stationid'], axis=1)
        self.xTest = self.xTest.drop(['datetime', 'stationid'], axis=1)


    def run_training(self):
        self.qrf = RandomForestQuantileRegressor(max_depth=12)
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
        # set foldername and create if necessary
        t = timenow()
        savedir = os.path.join(savedir, t)
        if not os.path.isdir(savedir):
            os.mkdir(savedir)
        # open file, load featuremap and close the data file
        self.xTest, map_shape = load_inference_data(datapath)

        # begin and time
        print('Predicting inference data....     ', end=' ')
        start_timer()
        try:
            self.yPred = self.qrf.predict(self.xTest, quantiles=[self.lowerCI, 0.5, self.upperCI])
        except AttributeError:
            try:
                self.yPred = self.qrf.predict(self.xTest, quantiles=[0.025, 0.5, 0.975])
            except AttributeError or ValueError as e:
                for key in self.xTest.keys():
                    print(f'Feature {key} \n\t min = {np.min(self.xTest[key])}\n\t max = {np.max(self.xTest[key])}')
                    if np.any(np.isinf(self.xTest[key])):
                        print(key)
                    if np.any(np.isnan(self.xTest[key])):
                        print(key)
                raise e

        end_timer()

        # restore original map shape
        prediction_map = reshape_preds(self.yPred, map_shape)

        # set .json save path and save output
        outputpath = os.path.join(savedir, t)
        tic = time.perf_counter()
        save_object(outputpath, prediction_map)
        toc = time.perf_counter()
        print(f'    save time {toc-tic:0.2f} seconds')

        # generate images
        if img:
            print('Generating images...')
            imgdir = os.path.join(savedir, f'TempMaps')
            if not os.path.isdir(imgdir):
                os.mkdir(imgdir)
                
            self.generate_images(prediction_map, imgdir)

        return savedir


    def run_validation(self, datapath, measurementpath, palmpath, resultpath=None, run_inference=True, generate_imgs=False):
        """
        Performs a validation run. A validation run consists of loading the feature maps, performing inference and
        evaluating the resulting temperature maps w.r.t. the moving average feature and analysing the accuracy per
        measurement station.
        """
        if run_inference:
            assert os.path.isdir(resultpath), 'If inference is to be run, resultpath must be the desired save path'
            resultpath = self.run_inference(datapath, resultpath, img=generate_imgs)
        else:
            assert os.path.isfile(resultpath), 'If inference is not run, a path to inference results must be given'
        # load boundary
        boundary = load_file(f'{os.path.splitext(palmpath)[0]}_boundary.z')
        # infopath = 'Data/stations.csv'
        infopath = os.path.join(os.path.dirname(os.path.dirname(measurementpath)), 'stations.csv')
        validation_evaluation(resultpath, datapath, boundary, infopath, measurementpath)


    @staticmethod
    def generate_images(inferencedata, imgpath, load=False):
        """
        Generates images suitable for loading into SR_GAN for training or testing. Note that the inference file loaded
        contains three prediction values: [CI lower bound, mean prediction, CI upper bound]. Currently only the mean
        prediction value is used to generate images.
        Option to pass data location instead of prediction maps for inferencedata parameter, in which case the load
        flag must be set to True.

        Normalisation: normalised using (nan)max/min temp values
        """
        if load:
            inferencedata = load_file(inferencedata)

        vmin = np.nanmin(inferencedata)
        vmax = np.nanmax(inferencedata)

        for time in range(inferencedata.shape[0]):
            tempmap = sns.heatmap(inferencedata[time, :, :, 1], vmin=vmin, vmax=vmax, linewidth=0,
                                  yticklabels=False, xticklabels=False)
            plt.close()
            fig = tempmap.get_figure()
            fig.savefig(os.path.join(imgpath, f'tempmap_{time}.png'), bbox_inches='tight', pad_inches=0)


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
            print(f'    Gathering error differences...........')
            nvar = len(variables)
            for idx, variable in enumerate(variables):
                print(f'      variable {idx}/{nvar} ({variable})')
                xtrain = self.xTrain.copy()
                var_oob_errors = []
                for i in range(n):
                    print(f'        run {i+1}', end='\r')
                    xtrain[variable] = sklearn.utils.shuffle(self.xTrain[variable]).values
                    var_oob_preds = self.qrf.predict(xtrain, oob_score=True)
                    var_oob_errors.append(np.mean(np.abs(var_oob_preds - self.yTrain)))
                print(end='\x1b[2k')
                oob_errors[variable] = np.mean(var_oob_errors) - og_oob_error
                diffsum += oob_errors[variable]

            print(f'    Calculating variable importances.......')
            for variable in variables:
                self.variable_importance['Variable'].append(variable)
                self.variable_importance['Importance [%]'].append(np.round((oob_errors[variable] / diffsum)*100, 2))

            modelname = os.path.splitext(os.path.basename(filepath))[0]
            savepath = os.path.join(os.path.dirname(filepath), modelname, f'{modelname}_variable_importance.csv')
            DataFrame(self.variable_importance).to_csv(savepath, index=False)


    def save_ouput(self, savedir, inference=False):
        output_df = self.output_file()
        if inference:
            if not os.path.isdir(os.path.dirname(savedir)):
                os.mkdir(os.path.dirname(savedir))
            output_df.to_csv(f'{savedir}.csv', index=False)
            print(f'  Inference output saved to {os.path.dirname(savedir)}')
        else:
            if not os.path.isdir(savedir):
                os.mkdir(savedir)
            output_df.to_csv(os.path.join(savedir, f'{self.modelname}.csv'), index=False)
            print(f'  Test output saved to {savedir}')


    def save_model(self, modelpath):
        joblib.dump(self, f'{modelpath}.z', compress=3)
        print(f'  Model {self.modelname} saved to {os.path.dirname(modelpath)}')


    def output_file(self):
        output = {'datetime': self.test_times}
        for featurekey in self.xTest.keys():
            output[featurekey] = self.xTest[featurekey]
        output['Prediction'] = self.yPred
        output['True Temperature'] = self.yTest
        output['stationid'] = self.test_stations

        for key in output.keys():
            output[key] = list(output[key])

        return DataFrame(output)
