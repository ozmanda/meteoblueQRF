import os
import numpy as np
from warnings import warn
from pandas import DataFrame, concat
from qrf_utils import start_timer, end_timer, sd_mu
import qrf_utils
from quantile_forest import RandomForestQuantileRegressor
import joblib
import matplotlib.pyplot as plt
from seaborn import histplot, scatterplot
from lcz_analysis import lcz_analysis


class DropsetQRF:
    def __init__(self, datasets, confidence_interval=None):
        """
        Class for QRF error estimation using the "dropset" method, similar to leave-one-out cross validation or
        k-fold cross validation. The data is split into training and test sets, with the test set being a single station.

        Args:
            datasets (_type_): dropset dataset, delivered as a dictionary containing a DataFrame for each station
            confidence_interval (_type_, optional): confidence interval for testing, must be between 0 and 1. Defaults to None.
        """
        self.Output = {}
        self.station_summary = {'Station': [], 'MSE': [], 'Standard Deviation': [], 'Mean Error': []}
        self.data = datasets
        self.stations = datasets.keys()
        CI: float = confidence_interval if confidence_interval else 95
        self.lowerCI: float = ((100 - CI) / 2)/100
        self.upperCI: float = (100 - self.lowerCI)/100
        self.aggregated_errors: DataFrame = DataFrame()

    def xy_generation(self, testkey: str):
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

    def run_error_estimation(self, savepath=None):
        if savepath:
            if not os.path.isdir(savepath):
                os.mkdir(savepath)
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
            if savepath:
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
            stationData[f'{self.lowerCI}%'] = yPred[:, 0]
            stationData[f'{self.upperCI}%'] = yPred[:, 2]
            stationData['Deviation'] = yTest - yPred[:, 1]
            stationData['Absolute Deviation'] = np.abs(stationData['Deviation'])
            stationData['MSE'] = np.sum(stationData['Absolute Deviation'] ** 2) / len(stationData['Absolute Deviation'])
            sd, mu = sd_mu(stationData['Deviation'])
            stationData['SD'] = sd
            stationData['ME'] = mu

            # enter station dictionary into output dictionary
            self.Output[key] = stationData

    def save_output(self, savepath):
        errorpath = os.path.join(savepath, 'errors')
        if not os.path.isdir(savepath):
            os.mkdir(savepath)
        if not os.path.isdir(errorpath):
            os.mkdir(errorpath)
        for key in self.Output:
            self.station_summary['Station'].append(key)
            self.station_summary['MSE'].append(self.Output[key].pop('MSE'))
            self.station_summary['Standard Deviation'].append(self.Output[key].pop('SD'))
            self.station_summary['Mean Error'].append(self.Output[key].pop('ME'))
            DataFrame(self.Output[key]).to_csv(os.path.join(errorpath, f'errors_{key}.csv'), index=False)

        summary = DataFrame(self.station_summary)
        summary.to_csv(os.path.join(savepath, 'station_summary.csv'), index=False)

        # sd and rmse
        with open(os.path.join(savepath, 'summary.txt'), 'w') as file:
            file.writelines([f'Standard Deviation: {np.mean(summary["Standard Deviation"])}',
                             f'Mean Error: {np.mean(summary["Mean Error"])}'])
            file.close()

    def generate_images(self, savepath):
        """
        Generate graphs and images for the dropset run as a whole and
        :param savepath: dropset save folder
        :return: none
        """
        imgdir = os.path.join(savepath, 'pred_vs_true')
        dists = os.path.join(savepath, 'distributions')

        # summary and per station graphs
        self.summary_graphs(savepath)
        for key in self.Output:
            self.pred_vs_true(os.path.join(imgdir, f'{key}.png'), self.Output[key])
            self.error_hist(os.path.join(dists, f'{key}.png'), self.Output[key]['Deviation'],
                            f'Prediction error distribution for station {key}')
            cleaned_devs, _ = qrf_utils.pop_outliers_std(DataFrame(self.Output[key]), 'Deviation')
            self.error_hist(os.path.join(dists, f'{key}_cleaned.png'), cleaned_devs['Deviation'],
                            f'Cleaned prediction error distribution for station {key}')

    def summary_graphs(self, path):
        """
        Generates graphs of the dropset run as a whole - all errors from all station and mean error from all stations
        :param path: path where the images are to be saved
        :return: None
        """
        dropset_errors = {'error': [], 'datetime': []}
        for key in self.Output.keys():
            dropset_errors['error'].append(self.Output[key]['Deviation'])
            dropset_errors['datetime'].append(self.Output[key]['datetime'])
        dropset_errors = DataFrame(dropset_errors)
        dropset_clean, dropset_outliers = qrf_utils.pop_outliers_std(dropset_errors, 'error')

        self.error_hist(dropset_errors, os.path.join(path, 'all_station_errors.png'),
                        'Dropset error distribution over all stations')
        self.error_hist(dropset_clean, os.path.join(path, 'all_station_errors_cleaned.png'),
                        'Dropset error distribution over all station')

        station_errors = DataFrame(self.station_summary)
        station_cleaned, station_outliers = qrf_utils.pop_outliers_std(station_errors, 'Mean Error')

        self.error_hist(station_errors, os.path.join(path, 'average_station_errors.png'),
                        'Average station dropset error distribution')
        self.error_hist(station_cleaned, os.path.join(path, 'average_station_errors.png'),
                        'Average station dropset error distribution')

    @staticmethod
    def error_hist(data, savepath, title):
        if not os.path.isfile(savepath):
            fig = histplot(data, x='error', stat='probability')
            fig.set_title(title)
            fig.set_xlabel('Prediction error [°C]')
            fig.set_ylabel('Probability')
            plt.xticks(rotation=45)
            plt.savefig(savepath, bbox_inches='tight')

    @staticmethod
    def pred_vs_true(path, data):
        if not os.path.isfile(path):
            plot = scatterplot(data, x='datetime', y='Predicted Temperature', color='r', label='Predicted Temperature')
            plt.plot(data, x='datetime', y='True Temperature', label='True Temperature')
            plot.set_title('Prediction vs. True Temperature')
            plot.set_xlabel('Time')
            plt.legend()
            plt.xticks(rotation=45)
            plt.savefig(path, bbox_inches='tight')

    def run_dropset_estimation(self, savepath, savemodels=True):
        self.run_error_estimation(os.path.join(savepath, 'models') if savemodels else None)
        self.save_output(savepath)
        self.generate_images(savepath)
        lcz_analysis(savepath)
