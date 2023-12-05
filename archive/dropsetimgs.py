import os
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv, to_datetime, DataFrame
from seaborn import scatterplot, histplot
from qrf_utils import sd_mu, mse


def pred_vs_true(path, file):
    if not os.path.isfile(path):
        dt = to_datetime(file['datetime'])
        plot = scatterplot(x=dt, y=file['Predicted Temperature'], color='r', label='Predicted Temperature')
        plt.plot(dt, file['True Temperature'], label='True Temperature')
        plot.set_title('Prediction vs. True Temperature')
        plot.set_xlabel('Time')
        plt.legend()
        plt.savefig(path)
        plt.close()


def dist(path, errors):
    if not os.path.isfile(path):
        errors = np.ravel(errors).astype(list)
        fig = histplot(errors)
        fig.set_title('Prediction Error Distribution')
        fig.set_xlabel('Prediction Error [°C]')
        plt.savefig(path)
        plt.close()


def extract_LCZ(stationscsv, stations_list):
    stations_dict = {}
    for _, row in stationscsv.iterrows():
        stations_dict[row["stationid_new"]] = row["classification"]
    LCZs = []
    for station in stations_list:
        LCZs.append(stations_dict[station])
    return LCZs


inferencefiles = 'DATA/QRF_Dropset_Results/run1'
imgdir = os.path.join(inferencefiles, 'pred_vs_true')
stationscsv = 'DATA/stations.csv'
if not os.path.isdir(imgdir):
    os.mkdir(imgdir)
dists = os.path.join(inferencefiles, 'distributions')
if not os.path.isdir(dists):
    os.mkdir(dists)

summary = {'Station': [], 'Standard Deviation': [], 'Mean Error': [], 'RMSE': [], 'LCZ': []}

for filename in os.listdir(os.path.join(inferencefiles, 'errors')):
    filepath = os.path.join(inferencefiles, 'errors', filename)
    if os.path.isfile(filepath):
        station = os.path.splitext(filename)[0].split('_')[1]
        file = read_csv(filepath, delimiter=',')
        pred_vs_true(os.path.join(imgdir, f'{station}.png'), file)
        dist(os.path.join(dists, f'{station}.png'), file['Deviation'])
        sd, mu = sd_mu(file['Deviation'])
        rmse = np.sqrt(mse(file['True Temperature'], file['Predicted Temperature']))
        summary['Station'].append(station)
        summary['Standard Deviation'].append(sd)
        summary['Mean Error'].append(mu)
        summary['RMSE'].append(rmse)

summary['LCZ'] = extract_LCZ(read_csv(stationscsv, delimiter=';'), summary['Station'])
df = DataFrame(summary)
df.to_csv(os.path.join(inferencefiles, 'station_summary.csv'), sep=',', index=False)

