import os
import utils
import joblib
import argparse
from QRF import QRF
from warnings import warn
from DropsetQRF import DropsetQRF


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', help='Type of QRF: "training" for normal QRF model training and testing or "dropset"'
                                       'for error estimation using dropset method, "evaluation" for the evaluation of'
                                       'pretrained models', default=None)
    parser.add_argument('--stationDatapath', help='Relative path to folder containing station data',
                        default='Data/MeasurementFeatures_v6')
    parser.add_argument('--starttime', help='Date and time of the beginning of the data interval in the format'
                                            'YYYY/MM/DD_HH:MM', default=None, nargs="*", type=str)
    parser.add_argument('--endtime', help='Date and time of the end of the data interval in the format'
                                          'YYYY/MM/DD_HH:MM', default=None, nargs="*", type=str)
    parser.add_argument('--test_start', help='Date and time of the beginning of test/inference interval in the format'
                                             'YYYY/MM/DD_HH:MM', default=None, nargs="*", type=str)
    parser.add_argument('--test_end', help='Date and time of the beginning of test/inference interval in the format'
                                           'YYYY/MM/DD_HH:MM', default=None, nargs="*", type=str)
    parser.add_argument('--savedir', help='Relative path to the save directory for QRF output (new folder will be made)',
                        default='Data')
    parser.add_argument('--modeldir', default=None, help='Path to directory for trained models')
    parser.add_argument('--CI', default=95, help='Confidence interval in percent (i.e. 95 for the 95% CI)')
    parser.add_argument('--modelname', default=None, help='Name of the model to be evaluated')
    parser.add_argument('--nestimators', default=None, help='Number of OOB predictions used to estimate varaible '
                                                            'importance')
    args = parser.parse_args()

    # ASSERT REQUIRED PARAMETERS
    assert args.type, 'A training type must be given'

    # QRF TRAINING RUN
    if args.type == "training":
        assert args.savedir, 'Directory for saving QRF output is required'
        assert args.modeldir, 'A path must be given for model saving'
        if args.starttime:
            assert args.endtime, 'If start time(s) for training is/are given, an end time must be given as well'
            if len(args.starttime) != len(args.endtime):
                warn(f'Number of start and end times for training set cannot be matched', UserWarning)
                raise ValueError
        if args.test_start:
            assert args.test_end, 'If start time(s) for testing is/are given, an end time must be given as well'
            if len(args.test_start) != len(args.test_end):
                warn(f'Number of start and end times for test set cannot be matched', UserWarning)
                raise ValueError

        qrf = QRF()
        # QRF run with one shuffled time window
        if not args.test_start and not args.test_end:
            dataset = utils.load_data(os.path.join(os.getcwd(), args.stationDatapath))
            qrf = QRF()
            qrf.set_split_data(dataset)

        # QRF run with specific training and test time windows
        else:
            dataset_train = utils.load_data(os.path.join(os.getcwd(), args.stationDatapath),
                                            startDatetime=args.starttime, endDatetime=args.endtime)
            assert len(dataset_train) != 0, 'No data found in training window'
            dataset_test = utils.load_data(os.path.join(os.getcwd(), args.stationDatapath),
                                           startDatetime=args.test_start, endDatetime=args.test_end)
            assert len(dataset_test) != 0, 'No data found in test window'

            qrf.set_data(dataTrain=dataset_train, dataTest=dataset_test)

        qrf.run_training()
        qrf.run_test()
        qrf.save_ouput(os.path.join(os.getcwd(), args.savedir), args.modeldir)

    # DROPSET ERROR ESTIMATION
    if args.type == 'dropset':
        assert args.savedir, 'Directory for saving QRF output is required'
        if args.starttime:
            assert args.endtime, 'If start time(s) is/are given, an end time must be given as well'
            if len(args.starttime) != len(args.endtime):
                warn(f'Number of start and end times for dropset cannot be matched', UserWarning)
                raise ValueError

        datasets = utils.load_data(os.path.join(os.getcwd(), args.stationDatapath),
                                   startDatetime=args.starttime, endDatetime=args.endtime, dropset=True)
        dropsetQRF = DropsetQRF(datasets, args.CI)
        dropsetQRF.run_error_estimation()
        dropsetQRF.save_output(os.path.join(os.getcwd(), args.savedir))

    # INFERENCE
    if args.type == 'inference':
        assert args.modeldir, 'Model directory must be given'
        assert args.modelname, 'Model name must be given for inference'
        dataset = utils.load_data(args.stationDatapath)
        qrf = joblib.load(os.path.join(args.modeldir, args.modelname))
        qrf.run_inference(dataset)


    # VARIABLE IMPORTANCE ANALYSIS
    if args.type == 'evaluation':
        assert args.modelname, 'Model name flag must be given for model evaluation'
        assert args.modeldir, 'Relative path to trained model directory must be given'

        # load trained model and run variable importance analysis
        qrf = joblib.load(os.path.join(args.modeldir, args.modelname))
        qrf.run_variable_importance_estimation(args.modeldir, args.modelname)

