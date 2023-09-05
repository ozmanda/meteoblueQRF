import os
import argparse
import numpy as np
from seaborn import heatmap
import matplotlib.pyplot as plt
from qrf_utils import load_file, save_object, reshape_preds, unravel_data
from datetime import datetime
import time
import joblib

"""
Quick analysis to check if LCZ or other model features are causing the "kacheln" we get during inference. 
!! Changing only inference data & loading pre-trained models for efficiency (kacheln could still occur during training)
"""

def load_inference_data(datapath, ):
    data = load_file(datapath)
    _ = data.pop('datetime')
    _ = data.pop('time')
    _ = data.pop('temperature')
    if 'moving average' in data.keys():
        data['moving_average'] = data['moving average']
        _ = data.pop('moving average')
    data = var_replacement(data)
    return data, data['moving_average'].shape


def var_replacement(xTest, variable='moving_average', type='uniform'):
    '''
    Random temp replacement for moving average temperature with three method options: either fill using a uniform value,
    a gradient from lowest to highest value or a four-square pattern.
    '''
    shape = xTest[variable].shape
    y = shape[1]
    x = shape[2]
    replacement = np.zeros(shape)
    if type == 'uniform':
        replacement = np.full(shape, 30)
    elif type == 'gradient':
        max = np.nanmax(xTest[variable])
        step = (max-min)/y
        for row in range(y):
            replacement[:, row] = [min + (step*row)]*x
    elif type == 'squares':
        xmid = int(x/2)
        ymid = int(y/2)
        max = np.nanmax(xTest[variable])
        min = np.nanmin(xTest[variable])
        replacement[:, :xmid, :ymid] = min
        replacement[:, xmid:, :ymid] = min + (max-min)/3
        replacement[:, :xmid, ymid:] = min + ((max-min)/3)*2
        replacement[:, xmid:, ymid:] = min + ((max-min)/3)*3

    xTest[variable] = replacement
    return xTest


def shorten(xTest):
    for key in xTest.keys():
        xTest[key] = xTest[key][10]
    return xTest


def map_vis(path: str, val_array: np.ndarray):
    '''
    Generates images of the given ndarray over all times (presumes 3 dimensions: lat, lon and time)
    '''
    if not os.path.isdir(path):
        os.mkdir(path)
    for time in range(val_array.shape[0]):
        savepath = os.path.join(path, f'time_{time}.png')
        ax = heatmap(val_array[time, :, :])
        plt.show()
        plt.savefig(savepath, bbox_inches='tight')
        plt.close()


def run_inference(qrf, xTest, savedir, type):
    print('Predicting inference data....     ')
    yPred = qrf.predict(xTest)

    prediction_map = reshape_preds(yPred, map_shape)
    savedir = os.path.join(savedir, f'{type}.json')

    save_object(savedir, prediction_map)
    savedir = os.path.join(savedir, 'imgs')
    map_vis(savedir, prediction_map)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--modelpath')
    parser.add_argument('--inferencedata')
    args = parser.parse_args()

    # create savedir if it does not already exist
    savedir = os.path.join('DATA', 'QRF_Inference_Results', args.name, 'kacheln_analysis')
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    # load pretrained qrf model
    print('Loading trained QRF model')
    qrf = joblib.load(args.modelpath)

    # open file, load featuremap and close the data file
    xTest, map_shape = load_inference_data(args.inferencedata)
    tests = ['uniform', 'gradient', 'squares']
    for test_type in tests:
        savedir_test = os.path.join(savedir, f'{test_type}')
        if not os.path.isdir(savedir_test):
            os.mkdir(savedir_test)
        xTest_new = var_replacement(xTest, variable='moving_average', type=test_type)
        xTest_new, _ = unravel_data(xTest_new)
        xTest_new = shorten(xTest_new)
        run_inference(qrf.qrf, xTest_new, savedir_test, test_type)
