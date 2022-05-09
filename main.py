import argparse
import utils
from DropsetQRF import DropsetQRF




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stationDatapath', help='absolute path to folder containing station data',
                        default='C://Users//ushe//PycharmProjects//Stadtklima QRF//Data//MeasurementFeatures_v6')
    parser.add_argument('--savefolder', help='absolute path to the desired save folder, which will be created if it '
                                             'does not already exist',
                        default='C://Users//ushe//PycharmProjects//pythonQRF//Data//DropsetData')
    parser.add_argument('--starttime', help='Date and time of the beginning of the data interval in the format'
                                            'YYYY/MM/DD HH:MM', default='2019/06/25 00:00')
    parser.add_argument('--endtime', help='Date and time of the end of the data interval in the format'
                                          'YYYY/MM/DD HH:MM', default='2019/06/29 00:00')
    parser.add_argument('--CI', help='Confidence interval in percent (i.e. 95 for the 95% CI', default=95)

    args = parser.parse_args()
    datasets = utils.load_data(args.stationDatapath, startDatetime=args.starttime, endDatetime=args.endtime)
    dropsetQRF = DropsetQRF(datasets, args.CI)
    dropsetQRF.run_error_estimation()
    dropsetQRF.save_output(args.savefolder)

