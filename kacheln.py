import os
import argparse
import numpy as np
from seaborn import heatmap
import matplotlib.pyplot as plt
from qrf_utils import load_inference_data, save_object, reshape_preds
from datetime import datetime

"""
Quick analysis to check if LCZ or other model features are causing the "kacheln" we get during inference. 
!! Changing only inference data & loading pre-trained models for efficiency (kacheln could still occur during training)
"""

def rand_temp(xTest, variable='moving_average', type='uniform'):
    '''
    Random temp replacement for moving average temperature using (METHOD!!)
    '''
    shape = xTest[variable].shape
    y = shape[1]
    x = shape[2]
    if type == 'uniform':
        replacement = np.full(shape, 30)
    elif type == 'gradient':
        replacement = np.zeros(shape)
        max = np.nanmax(xTest[variable])
        step = (max-min)/y
        for row in range(y):
            replacement[:, row] = [min + (step*row)]*x
    elif type == 'squares':
        replacement = np.zeros(shape)
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


def map_vis(path: str, val_array: np.ndarray):
    '''
    Generates images of the given ndarray over all times (presumes 3 dimensions: lat, lon and time)
    '''
    if not os.path.isdir(path):
        os.mkdir(path)
    for time in range(val_array.shape[0]):
        savepath = os.path.join(path, f'errormap_t.{time}.png')
        if not os.path.isfile(savepath):
            print('    error heatmap')
            ax = heatmap(val_array[time, :, :])
            plt.show()
            plt.savefig(savepath, bbox_inches='tight')
            plt.close()


def run_inference(datapath, savedir, img=True):
        # open file, load featuremap and close the data file
        xTest, map_shape = load_inference_data(datapath)

        # begin and time
        print('Predicting inference data....     ')
        yPred = qrf.predict(xTest)

        prediction_map = reshape_preds(yPred, map_shape)
        timenow = datetime.now().replace(second=0, microsecond=0)
        timenow = f'{timenow.year}-{timenow.month}-{timenow.day}_{timenow.hour}.{timenow.minute}_ma'
        savedir = os.path.join(savedir, f'{timenow}.json')

        save_object(savedir, prediction_map)
        if img:
            print('Generating images...')
            savedir = os.path.join(os.path.dirname(savedir), timenow, 'prediction_maps')
            map_vis(prediction_map, savedir)

        return savedir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelpath')
    parser.add_argument('--savedir')
    parser.add_argument('--inferencedata')
    parser.add_argument('--imgpath')

    # create savedir if it does not already exist
    if not os.path.isdir(args.savedir):
        os.mkdir(args.savedir)

    # load pretrained qrf model
    print('Loading trained QRF model')
    tic = time.perf_counter()
    qrf = joblib.load(args.modeldir)
    toc = time.perf_counter()
    savedir = qrf.run_inference(args.inferencedata, args.savedir)
    print(f'Inference file saved at: {savedir}')