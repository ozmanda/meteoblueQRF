import os
import joblib
import numpy as np
from seaborn import heatmap
import matplotlib.pyplot as plt
import imageio

featurepath = 'DATA/QRF_Inference_Feature_Maps/mb_4_multi_stations_xy_N02.00m_palmhumi.json'
savepath = 'DATA/QRF_Inference_Feature_Maps/mb_4_multi_stations_xy_N02.00m_palmhumi_imgs'
if not os.path.isdir(savepath):
    os.mkdir(savepath)
featurename = 'moving_average'
savepath = os.path.join(savepath, featurename)
if not os.path.isdir(savepath):
    os.mkdir(savepath)
featuremap = joblib.load(featurepath)
times = featuremap['datetime']
featuremap = featuremap[featurename]

for time in range(featuremap.shape[0]):
    path = os.path.join(savepath, f'{featurename}_{time}.png')
    ax = heatmap(featuremap[time, :, :], vmax=np.nanmax(featuremap), vmin=np.nanmin(featuremap))
    ax.set_title(times[time, 0, 0])
    plt.show()
    plt.savefig(path, bbox_inches='tight')
    plt.close()

imgs = []
for filename in os.listdir(savepath):
    imgs.append(imageio.imread(os.path.join(savepath, filename)))
imageio.mimsave(os.path.join(savepath, f'{featurename}.gif'), imgs)
