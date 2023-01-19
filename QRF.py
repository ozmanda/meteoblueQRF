import os
import joblib
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from pandas import DataFrame, concat
from utils import start_timer, end_timer, mse
from datetime import datetime
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

    def save_model(self, modelpath):
        timenow = datetime.now().replace(second=0, microsecond=0)
        timenow = f'{timenow.year}-{timenow.month}-{timenow.day}_{timenow.hour}.{timenow.minute}'
        joblib.dump(self.qrf, os.path.join(modelpath, f'{timenow}_{self.MSE}.z'),
                    compress=3)

    def save_ouput(self, savedir, modelpath):
        self.save_model(modelpath)
        if not os.path.isdir(savedir):
            os.mkdir(savedir)
        savedir = os.path.join(savedir, f'{datetime.now().replace(second=0, microsecond=0)}_{self.MSE}.csv')

        output = {}
        output['datetime'] = self.test_times
        for featurekey in self.xTest.keys():
            output[featurekey] = self.xTest[featurekey]
        output['Prediction'] = self.yPred
        output['True Temperature'] = self.yTest

        for key in output.keys():
            output[key] = list(output[key])
        output_df = DataFrame(output)

        # save data output and model
        output_df.to_csv(savedir, index=False)
