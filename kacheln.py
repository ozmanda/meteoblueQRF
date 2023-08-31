import os
import argparse
from qrf_utils import load_inference_data

"""
Quick analysis to check if LCZ or other model features are causing the "kacheln" we get during inference. 
!! Changing only inference data & loading pre-trained models for efficiency (kacheln could still occur during training)
"""

def rand_temp(xTest):
    '''
    Random temp replacement for moving average temperature using (METHOD!!)
    '''

    return 5


def run_inference(datapath, savedir, img=True):
        # open file, load featuremap and close the data file
        xTest, map_shape = load_inference_data(datapath)

        # begin and time
        print('Predicting inference data....     ')
        yPred = qrf.predict(xTest)

        prediction_map = qrf_utils.reshape_preds(yPred, map_shape)
        timenow = datetime.now().replace(second=0, microsecond=0)
        timenow = f'{timenow.year}-{timenow.month}-{timenow.day}_{timenow.hour}.{timenow.minute}_ma'
        savedir = os.path.join(savedir, f'{timenow}.json')

        save_object(savedir, prediction_map)
        if img:
            print('Generating images...')
            savedir = os.path.join(os.path.dirname(savedir), timenow, 'prediction_maps')
            qrf_utils.map_vis(prediction_map, savedir)

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