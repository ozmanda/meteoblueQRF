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
                        default='~/meteoblue/DATA/Measurement_Datasets/run3/')
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
    parser.add_argument('--savedir', help='Relative path to the save directory for QRF output (new folder will be made).'
                                          'For inference, include the desired filename', default=None)
    parser.add_argument('--infopath', default='', help='Path to file containing station information for LCZ analysis')
    parser.add_argument('--modelpath', default=None, help='Save path for trained models or path to model for '
                                                         'inference or evaluation (including model name)')
    parser.add_argument('--savemodels', default=True, help='Indicates if individual dropset models should be saved.')
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
        assert os.path.isdir(os.path.dirname(args.modelpath)), 'A path must be given for model saving'
        if args.starttime:
            assert args.endtime, 'If start time(s) for training is/are given, an end time must be given as well'
            if len(args.starttime) != len(args.endtime):
                warn(f'Number of start and end times for training set cannot be matched', UserWarning)
                raise ValueError
            if not args.test_start or not args.test_end:
                args.test_start, args.test_end = qrf_utils.set_test_times(args.starttime, args.endtime)

        if args.test_start:
            assert args.test_end, 'If start time(s) for testing is/are given, an end time must be given as well'
            if len(args.test_start) != len(args.test_end):
                warn(f'Number of start and end times for test set cannot be matched', UserWarning)
                raise ValueError

        qrf = QRF(modelname=os.path.basename(args.modelpath) ,confidence_interval=args.CI)
        qrf.load_training_data(args.stationDatapath, start=args.starttime, end=args.endtime)
        qrf.load_test_data(args.stationDatapath, start=args.test_start, end=args.test_end)
        qrf.run_training()
        qrf.run_test()
        qrf.save_model(args.modelpath)
        qrf.save_ouput(args.savedir)
        qrf.save_trainingset(args.savedir)

    # DROPSET ERROR ESTIMATION
    elif args.type == 'dropset':
        # create savedir if it does not already exist
        assert os.path.isdir(args.stationDatapath)
        assert args.savedir, 'Directory for saving QRF dropset output is required'
        assert args.infopath, 'infopath must be given for LCZ extraction'
        if not os.path.isdir(args.savedir):
            os.mkdir(args.savedir)

        if args.starttime:
            assert args.endtime, 'If start time(s) is/are given, an end time must be given as well'
            if len(args.starttime) != len(args.endtime):
                warn(f'Number of start and end times for dropset cannot be matched', UserWarning)
                raise ValueError

        datasets = qrf_utils.load_dropset_data(os.path.join(os.getcwd(), args.stationDatapath),
                                       startDatetime=args.starttime, endDatetime=args.endtime)
        dropsetQRF = DropsetQRF(datasets, args.infopath, args.CI)
        dropsetQRF.run_dropset_estimation(args.savedir, savemodels=args.savemodels)

    # INFERENCE
    elif args.type == 'inference':
        assert os.path.isfile(args.modelpath), 'Model path must be given for inference'
        assert args.savedir, 'Directory for saving QRF output is required'
        assert os.path.isdir(args.stationDatapath), 'Data must be given for inference'
        assert args.test_start, 'Start time for inference must be given'
        assert args.test_end, 'End time for inference must be given'

        # create savedir if it does not already exist
        if not os.path.isdir(args.savedir):
            os.mkdir(args.savedir)

        # load pretrained qrf model
        print('Loading trained QRF model')
        tic = time.perf_counter()
        qrf: QRF = joblib.load(args.modelpath)
        toc = time.perf_counter()
        qrf.load_test_data(args.stationDatapath, start=args.test_start, end=args.test_end)
        print(f'    loading time: {toc-tic:0.4f} seconds\n')
        qrf.run_test()
        qrf.save_ouput(args.savedir, inference=True)

    # VALIDATION RUN AND RESULT EVALUATION
    elif args.type == 'validation':
        assert os.path.isfile(args.modelpath), 'Model path must be a file'
        assert os.path.isfile(args.inferencedata), 'Data must be given for validation'
        assert os.path.isdir(args.stationDatapath), 'Path to station data must be given'
        assert os.path.isfile(args.palmpath), 'Path to PALM simulation file must be given'
        assert args.savedir, 'Either a save folder or an inference result file must be given'
        qrf = joblib.load(args.modelpath)
        if os.path.isfile(args.savedir):
            # passing a savedir will mean that validation has already been run and use the existing inference results
            assert os.path.isfile(args.savedir), 'Inference results must be an .json file or the flag left empty'
            qrf.run_validation(args.inferencedata, args.stationDatapath, args.palmpath,
                               resultpath=args.savedir, run_inference=False, generate_imgs=args.generate_images)
        else:
            qrf.run_validation(args.inferencedata, args.stationDatapath, args.palmpath, resultpath=args.savedir, generate_imgs=args.generate_images)

    # VARIABLE IMPORTANCE ANALYSIS
    elif args.type == 'evaluation':
        assert os.path.isfile(args.modelpath), 'Model path must be a file'

        # load trained model and run variable importance analysis
        print('Loading trained QRF model')
        qrf = joblib.load(args.modelpath)
        qrf.run_variable_importance_estimation(args.modelpath)
