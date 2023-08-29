import os
import time

import qrf_utils
import joblib
import argparse
from QRF import QRF
from warnings import warn
from DropsetQRF import DropsetQRF


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('type', help='Type of QRF: "training" for normal QRF model training and testing or "dropset"'
                                     'for error estimation using dropset method, "evaluation" for the evaluation of'
                                     'pretrained models', default=None)
    parser.add_argument('--stationDatapath', help='Relative path to folder containing station data',
                        default='Data/MeasurementFeatures_v6')
    parser.add_argument('--inferencedata', help='Path to json file containing feature map for inference/validation',
                        default=None, type=str)
    parser.add_argument('--starttime', help='Date and time of the beginning of the data interval in the format'
                                            'YYYY/MM/DD_HH:MM', default=None, nargs="*", type=str)
    parser.add_argument('--endtime', help='Date and time of the end of the data interval in the format'
                                          'YYYY/MM/DD_HH:MM', default=None, nargs="*", type=str)
    parser.add_argument('--test_start', help='Date and time of the beginning of test/inference interval in the format'
                                             'YYYY/MM/DD_HH:MM', default=None, nargs="*", type=str)
    parser.add_argument('--test_end', help='Date and time of the beginning of test/inference interval in the format'
                                           'YYYY/MM/DD_HH:MM', default=None, nargs="*", type=str)
    parser.add_argument('--savedir', help='Relative path to the save directory for QRF output (new folder will be made)',
                        default=None)
    parser.add_argument('--modeldir', default=None, help='Path to directory for trained models for training or path to '
                                                         'model for inference or evaluation')
    parser.add_argument('--savemodels', default=False, help='Indicates if individual dropset models should be saved.')
    parser.add_argument('--CI', default=95, help='Confidence interval in percent (i.e. 95 for the 95% CI)')
    parser.add_argument('--generate_images', type=bool, help='Boolean value indicating if images for use in SR_GAN'
                                                             'should be generated (True/False)', default=False)
    parser.add_argument('--imagepath', type=str, help='Path to where images should be stored. Default is None, a path'
                                                      'will be generated automatically if none is given', default=None)
    parser.add_argument('--palmpath', type=str, help='Path to PALM simulation file for validation evaluation', default=None)
    args = parser.parse_args()
    assert args.type, 'A training type must be given'

    # QRF TRAINING RUN
    if args.type == "training":
        assert os.path.isdir(args.savedir), 'Directory for saving QRF output is required'
        assert os.path.isdir(args.modeldir), 'A path must be given for model saving'
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

        qrf = QRF(confidence_interval=args.CI)
        # QRF run with one shuffled time window
        if not args.test_start:
            dataset = qrf_utils.load_data(args.stationDatapath)
            qrf.set_split_data(dataset)

        # QRF run with specific training and test time windows
        else:
            dataset_train = qrf_utils.load_data(args.stationDatapath,
                                                startDatetime=args.starttime, endDatetime=args.endtime)
            assert len(dataset_train) != 0, 'No data found in training window'
            dataset_test = qrf_utils.load_data(args.stationDatapath,
                                               startDatetime=args.test_start, endDatetime=args.test_end)
            assert len(dataset_test) != 0, 'No data found in test window'

            qrf.set_data(dataTrain=dataset_train, dataTest=dataset_test)

        qrf.run_training()
        qrf.run_test()
        qrf.save_ouput(os.path.join(os.getcwd(), args.savedir), args.modeldir)

    # DROPSET ERROR ESTIMATION
    elif args.type == 'dropset':
        # create savedir if it does not already exist
        assert os.path.isdir(args.stationDatapath)
        assert args.savedir, 'Directory for saving QRF dropset output is required'
        if not os.path.isdir(args.savedir):
            os.mkdir(args.savedir)

        if args.starttime:
            assert args.endtime, 'If start time(s) is/are given, an end time must be given as well'
            if len(args.starttime) != len(args.endtime):
                warn(f'Number of start and end times for dropset cannot be matched', UserWarning)
                raise ValueError

        datasets = qrf_utils.load_dropset_data(os.path.join(os.getcwd(), args.stationDatapath),
                                       startDatetime=args.starttime, endDatetime=args.endtime)
        dropsetQRF = DropsetQRF(datasets, args.CI)
        dropsetQRF.run_dropset_estimation(args.savedir, savemodels=args.savemodels)

    # INFERENCE
    elif args.type == 'inference':
        assert os.path.isfile(args.modeldir), 'Model path must be given for inference'
        assert args.savedir, 'Directory for saving QRF output is required'
        assert os.path.isfile(args.inferencedata), 'Data must be given for validation'

        # create savedir if it does not already exist
        if not os.path.isdir(args.savedir):
            os.mkdir(args.savedir)

        # load pretrained qrf model
        print('Loading trained QRF model')
        tic = time.perf_counter()
        qrf = joblib.load(args.modeldir)
        toc = time.perf_counter()
        print(f'    loading time: {toc-tic:0.4f} seconds\n')
        savedir = qrf.run_inference(args.inferencedata, args.savedir)
        if args.generate_images:
            if not args.imagepath:
                imgpath = os.path.join('Data', 'QRF_Inference_Maps', f'{os.path.split(savedir)[1].split(".json")[0]}')
            else:
                imgpath = os.path.join(args.imagepath, f'{os.path.split(savedir)[1].split(".json")[0]}')
            if not os.path.isdir(imgpath):
                os.mkdir(imgpath)
            qrf.generate_images(savedir, imgpath)
        print(f'Inference file saved at: {savedir}')

    # VALIDATION RUN AND RESULT EVALUATION
    elif args.type == 'validation':
        assert os.path.isfile(args.modeldir), 'Model path must be a file'
        assert os.path.isfile(args.inferencedata), 'Data must be given for validation'
        assert os.path.isdir(args.stationDatapath), 'Path to station data must be given'
        assert os.path.isfile(args.palmpath), 'Path to PALM simulation file must be given'
        assert args.savedir, 'Either a save folder or an inference result file must be given'
        qrf = joblib.load(args.modeldir)
        if os.path.isfile(args.savedir):
            # passing a savedir will mean that validation has already been run and use the existing inference results
            assert os.path.isfile(args.savedir), 'Inference results must be an .json file or the flag left empty'
            qrf.run_validation(args.inferencedata, args.stationDatapath, args.palmpath,
                               resultpath=args.savedir, run_inference=False)
        else:
            qrf.run_validation(args.inferencedata, args.stationDatapath, args.palmpath, resultpath=args.savedir)

    # VARIABLE IMPORTANCE ANALYSIS
    elif args.type == 'evaluation':
        assert os.path.isfile(args.modeldir), 'Model path must be a file'

        # load trained model and run variable importance analysis
        qrf = joblib.load(args.modeldir)
        qrf.run_variable_importance_estimation(args.modeldir)

