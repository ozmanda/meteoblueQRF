import os
import argparse
import utils
from DropsetQRF import DropsetQRF
from warnings import warn
from QRF import QRF


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', help='Type of QRF: "training" for normal QRF model training and testing or "dropset"'
                                       'for error estimation using dropset method', default='training')
    parser.add_argument('--stationDatapath', help='Relative path to folder containing station data',
                        default='Data/MeasurementFeatures_v6')
    parser.add_argument('--starttime', help='Date and time of the beginning of the data interval in the format'
                                            'YYYY/MM/DD_HH:MM', default=['2019/06/25_00:00'], nargs="*", type=str)
    parser.add_argument('--endtime', help='Date and time of the end of the data interval in the format'
                                          'YYYY/MM/DD_HH:MM', default=['2019/06/29_23:59'], nargs="*", type=str)
    parser.add_argument('--test_start', help='Date and time of the beginning of test interval in the format'
                                          'YYYY/MM/DD_HH:MM', default=['2019/06/30_00:00'], nargs="*", type=str)
    parser.add_argument('--test_end', help='Date and time of the beginning of test interval in the format'
                                          'YYYY/MM/DD_HH:MM', default=['2019/06/30_23:59'], nargs="*", type=str)
    parser.add_argument('--savedir', help='Relative path to the save directory for QRF output (new folder will be made)',
                        default='Data')
    parser.add_argument('--modeldir', default=None, help='Path to directory for trained models')
    parser.add_argument('--DropsetSavefolder', help='Relative path to the desired save folder for the dropset error '
                                                    'data, which will be created if it does not already exist',
                        default='Data/DropsetData')
    parser.add_argument('--CI', help='Confidence interval in percent (i.e. 95 for the 95% CI)', default=95)

    args = parser.parse_args()

    # QRF TRAINING RUN
    if args.type == "training":
        if len(args.starttime) != len(args.endtime):
            warn(f'Number of start and end times for training set cannot be matched', UserWarning)
            raise ValueError
        elif len(args.test_start) != len(args.test_end):
            warn(f'Number of start and end times for test set cannot be matched', UserWarning)
            raise ValueError

        dataset_train = utils.load_data(os.path.join(os.getcwd(), args.stationDatapath),
                                        startDatetime=args.starttime, endDatetime=args.endtime)
        if len(dataset_train) == 0:
            warn('No data found for the given training times')
            raise ValueError

        dataset_test = utils.load_data(os.path.join(os.getcwd(), args.stationDatapath),
                                       startDatetime=args.test_start, endDatetime=args.test_end)

        qrf = QRF(dataset_train, dataset_test)
        qrf.run_training()

        if len(dataset_test) != 0:
            qrf.run_inference()
        else:
            warn('No data found for the given testing times')

        qrf.save_ouput(os.path.join(os.getcwd(), args.savedir), args.modeldir)

    # DROPSET ERROR ESTIMATION
    if args.type == 'dropset':
        if len(args.starttime) != len(args.endtime):
            warn(f'Number of start and end times for dropset cannot be matched', UserWarning)
            raise ValueError

        datasets = utils.load_data(os.path.join(os.getcwd(), args.stationDatapath),
                                   startDatetime=args.starttime, endDatetime=args.endtime, dropset=True)
        dropsetQRF = DropsetQRF(datasets, args.CI)
        dropsetQRF.run_error_estimation()
        dropsetQRF.save_output(os.path.join(os.getcwd(), args.savefolder))
