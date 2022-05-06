import argparse
import utils
from DropsetQRF import DropsetQRF




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stationDatapath', help='absolute path to folder containing station data',
                        default='C://Users//ushe//PycharmProjects//Stadtklima QRF//Data//MeasurementFeatures_v6')
    parser.add_argument('--starttime', help='Date and time of the beginning of the data interval in the format'
                                            'YYYY/MM/DD HH:MM', default='2019/06/25 00:00')
    parser.add_argument('--endtime', help='Date and time of the end of the data interval in the format'
                                          'YYYY/MM/DD HH:MM', default='2019/06/29 00:00')

    args = parser.parse_args()
    # datasets = utils.load_data(args.stationDatapath, startDatetime=args.starttime, endDatetime=args.endtime)
    # dropsetQRF = DropsetQRF(datasets)
    # errors = dropsetQRF.run_error_estimation()
    #

    from quantile_forest import RandomForestQuantileRegressor
    from sklearn import datasets

    x, y = datasets.fetch_california_housing(return_X_y=True)

    qrf = RandomForestQuantileRegressor()
    qrf.fit(x, y)

    yPred = qrf.predict(x, quantiles=[0.025, 0.5, 0.975])

    x = 5
