import argparse
import utils
from DropsetQRF import DropsetQRF
from QRF import QRF


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', help='Type of QRF: "model" for normal QRF model training and testing or "dropset"'
                                       'for error estimation using dropset method', default='model')
    parser.add_argument('--stationDatapath', help='absolute path to folder containing station data',
                        default='C://Users//ushe//PycharmProjects//Stadtklima QRF//Data//MeasurementFeatures_v6')
    parser.add_argument('--starttime', help='Date and time of the beginning of the data interval in the format'
                                            'YYYY/MM/DD_HH:MM', default='2019/06/25_00:00')
    parser.add_argument('--endtime', help='Date and time of the end of the data interval in the format'
                                          'YYYY/MM/DD_HH:MM', default='2019/06/29_23:59')
    parser.add_argument('--test_start', help='Date and time of the beginning of test interval in the format'
                                          'YYYY/MM/DD_HH:MM', default='2019/06/30_00:00')
    parser.add_argument('--test_end', help='Date and time of the beginning of test interval in the format'
                                          'YYYY/MM/DD_HH:MM', default='2019/06/30_23:59')
    parser.add_argument('--savedir', help='Absolute path to the save directory for QRF output (new folder will be made',
                        default='C://Users//ushe//PycharmProjects//pythonQRF//Data')
    parser.add_argument('--DropsetSavefolder', help='absolute path to the desired save folder for the dropset error '
                                                    'data, which will be created if it does not already exist',
                        default='C://Users//ushe//PycharmProjects//pythonQRF//Data//DropsetData')
    parser.add_argument('--CI', help='Confidence interval in percent (i.e. 95 for the 95% CI)', default=95)

    args = parser.parse_args()

    # QRF TRAINING RUN
    dataset_train = utils.load_data(args.stationDatapath, startDatetime=args.starttime, endDatetime=args.endtime)
    dataset_test = utils.load_data(args.stationDatapath, startDatetime=args.test_start, endDatetime=args.test_end)
    qrf = QRF(dataset_train, dataset_test)
    qrf.run_training()
    qrf.run_inference()
    qrf.save_ouput(args.savedir)

    # DROPSET ERROR ESTIMATION
    if args.type == 'dropset':
        datasets = utils.load_data(args.stationDatapath, startDatetime=args.starttime, endDatetime=args.endtime)
        dropsetQRF = DropsetQRF(datasets, args.CI)
        dropsetQRF.run_error_estimation()
        dropsetQRF.save_output(args.savefolder)
