import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import imageio
import joblib
import os
import re

def lcz_stats(df, lczs, dropset_run_path):
    stats = pd.DataFrame()
    stats['LCZ'] = lczs
    stats['Mean RMSE'] = [round(df[df['LCZ'] == lcz]["RMSE"].mean(), 4) for lcz in lczs]
    stats['Stations'] = [len(df[df['LCZ'] == lcz]) for lcz in lczs]
    stats['%'] = [round(len(df[df['LCZ'] == lcz]) / len(df) * 100, 2) for lcz in lczs]
    stats['Mean SD'] = [round(df[df['LCZ'] == lcz]["Standard Deviation"].mean(), 4) for lcz in lczs]
    stats.sort_values(by='Mean RMSE', ascending=False, inplace=True)
    stats.to_csv(os.path.join(dropset_run_path, 'lcz_summary.csv'), index=False)
    return stats

def split_lczs(lcz):
    lczs = [*re.sub(r'[^\w]', '', lcz)]
    if 'o' in lczs: lczs.remove('o')
    if 'r' in lczs: lczs.remove('r')
    return lczs

def lcz_data(stations, station_errors, lcz_dict, paths):
    stationinfo = pd.read_csv('S:/pools/t/T-IDP-Projekte-u-Vorlesungen/Meteoblue/Data/Messdaten/stations.csv', delimiter=';', index_col=False)
    stationinfo.dropna(inplace=True, subset='classification')
    stationinfo['classification'] = stationinfo['classification'].astype(str)
    stationinfo['classification'] = stationinfo['classification'].apply(lambda x: x[0])

    lcz_errors = {lcz: {'station': [], 'error': [], 'prediction': [], 'true': []} for lcz in lcz_dict.keys()}
    lcz_df = {'lcz': [], 'error': [], 'prediction': [], 'true': [], 'station': [], 'run': [], 'datetime': []}

    # gather all station errors of a certain LCZ
    for station in stations:
        try:
            lcz = stationinfo[stationinfo['stationid_new'] == station]['classification'].iloc[0]
        except IndexError:
            continue
        
        for run in range(len(paths)):
            run = f'{run+1}'
            try:
                l = len(station_errors[station][run]['Deviation'].values)
                lcz_df['lcz'].extend([lcz]*l)
                lcz_df['run'].extend([run]*l)
                lcz_df['station'].extend([station]*l)
                lcz_df['datetime'].extend(list(station_errors[station][run]['datetime'].values))
                lcz_df['error'].extend(list(station_errors[station][run]['Deviation'].values))
                lcz_df['prediction'].extend(list(station_errors[station][run]['Predicted Temperature'].values))
                lcz_df['true'].extend(list(station_errors[station][run]['True Temperature'].values))
            except KeyError:
                continue

    lcz_df = pd.DataFrame(lcz_df)
    lcz_df.loc[:, 'datetime'] = lcz_df['datetime'].apply(lambda x: pd.to_datetime(x))
    return lcz_df


def station_data(paths):
    station_summaries = []

    for path in paths:
        station_summaries.append(pd.read_csv(os.path.join(path, 'station_summary.csv'), delimiter=';', index_col=False))

    stations = []
    for i, station_summary in enumerate(station_summaries):
        stations.extend(station_summary['Station'].values)

    stations = np.unique(stations)

    station_errors = {station: {} for station in stations}
    errorpaths = [os.path.join(path, 'errors') for path in paths]
    for station in stations:
        station_errors[station]['All'] = pd.DataFrame()
        for i, errorpath in enumerate(errorpaths):
            run = f'{i+1}'
            try:
                station_errors[station][run] = pd.read_csv(os.path.join(errorpath, f'errors_{station}.csv'), index_col=False)
                station_errors[station]['All'] = pd.concat([station_errors[station]['All'], station_errors[station][run]])
            except FileNotFoundError:
                continue
    return station_summaries, station_errors, stations


def lcz_stat_graphs(station_summaries, lcz_dict, paths):
    for i in range(len(paths)):
        run = f'{i+1}'
        # remove secondary LCZ categorisations
        station_summaries[i]['LCZ'] = station_summaries[i]['LCZ'].astype(str)
        station_summaries[i]['LCZ'] = station_summaries[i]['LCZ'].apply(lambda x: x[0])
        # alternative: create a list of categorisations
        # df['LCZs'] = df['LCZ'].apply(split_lczs)
        
        available_lczs = list(station_summaries[i]['LCZ'].unique())
        available_lczs.remove('n')
        unavailable_lczs = set(lcz_dict.keys()) - set(available_lczs)

        print(f'Run {run}:\n\tavailable: {available_lczs}\n\tunavailable: {unavailable_lczs}')
        stats = lcz_stats(station_summaries[i], available_lczs, paths[i])

        # 3 plots: stations per lcz, rmse and sd
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax1 = sns.barplot(x='LCZ', y='Stations', data=stats, ax=ax[0], color='darkturquoise')
        ax1.set_xlabel('LCZ')
        ax1.set_ylabel('Number of stations')
        ax1.set_title('Number of stations per LCZ')

        ax2 = sns.barplot(x='LCZ', y='Mean RMSE', data=stats, ax=ax[1], color='darkturquoise')
        ax2.set_xlabel('LCZ')
        ax2.set_ylabel('Mean RMSE')
        ax2.set_title('Mean RMSE per LCZ')
        
        ax3 = sns.barplot(x='LCZ', y='Mean SD', data=stats, ax=ax[2], color='darkturquoise')
        ax3.set_xlabel('LCZ')
        ax3.set_ylabel('Mean SD')
        ax3.set_title('Mean SD per LCZ')

        fig.suptitle(f'Run {run}', fontsize=16)
        plt.savefig(os.path.join(paths[i], 'lcz_stats.png'), dpi=300)


def by_run_and_time(df):
    df.loc[:, 'datetime'] = df['datetime'].apply(lambda x: pd.to_datetime(x))
    runs = df['run'].unique()
    runs.sort()
    palmtimes = [[pd.to_datetime('2019-06-25 18:00:00'), pd.to_datetime('2019-06-27 02:00:00')],
                [pd.to_datetime('2019-08-03 18:00:00'), pd.to_datetime('2019-08-05 02:00:00')],
                [pd.to_datetime('2019-08-13 18:00:00'), pd.to_datetime('2019-08-15 02:00:00')],
                [pd.to_datetime('2019-08-17 18:00:00'), pd.to_datetime('2019-08-19 02:00:00')],
                [pd.to_datetime('2019-08-24 18:00:00'), pd.to_datetime('2019-08-26 02:00:00')]]

    for idx, run in enumerate(runs):
        df_run = df[df['run'] == run].copy()
        df_run['hour'] = df_run['datetime'].apply(lambda x: x.hour)

        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        plt.suptitle(f'Run {run}: {palmtimes[idx][0]} - {palmtimes[idx][1]}', fontsize=16)

        ax = sns.scatterplot(x='datetime', y='error', hue='hour', ax=axs[0], data=df_run, color='darkturquoise')
        ax.axhline(y=0.5, color='red')
        ax.axhline(y=-0.5, color='red')

        ax = sns.boxplot(x='hour', y='error', ax=axs[1], data=df_run, color='darkturquoise')
        ax.axhline(y=0.5, color='red')
        ax.axhline(y=-0.5, color='red')

def run_boxplot(lcz_df, lcz_cleaned, run):
    # Analysis of run 4
    run_df = lcz_df[lcz_df['run'] == run]
    run_df_cleaned = lcz_cleaned[lcz_cleaned['run'] == run]
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    ax1 = sns.boxplot(x='lcz', y='error', ax=axs[0], data=run_df, color='darkturquoise')
    ax1.axhline(y=0.5, color='red')
    ax1.axhline(y=-0.5, color='red')
    ax1.set_title('Uncleaned')
    ax2 = sns.boxplot(x='lcz', y='error', ax=axs[1], data=run_df_cleaned, color='darkturquoise')
    ax2.axhline(y=0.5, color='red')
    ax2.axhline(y=-0.5, color='red')
    ax2.set_title('Cleaned')


def run_lcz_scatterplot(outliers, lcz_dict, run, lczs):
    run_outliers = outliers[outliers['run'] == run]
    l = len(lczs)
    fig, axs = plt.subplots(round(l/2), 2, figsize=(20, 15), tight_layout=True)
    for idx, lcz in enumerate(lczs):
        ax = sns.scatterplot(x='datetime', y='error', ax=axs[int(idx/2), idx%2], 
                            data=run_outliers[run_outliers['lcz'] == lcz])
        ax.set_title(f'{lcz_dict[lcz]["Name"]} ({lcz})')
    return run_outliers


def feature_imgs(featurepath, featurename, cmap=None):
    savepath = os.path.join(savepath, featurename)
    if not os.path.isdir(savepath):
        os.mkdir(savepath)
    featuremap = joblib.load(featurepath)
    times = featuremap['datetime'].unique()
    featuremap = featuremap[featurename]

    for time in range(featuremap.shape[0]):
        path = os.path.join(savepath, f'{featurename}_{time}.png')
        ax = sns.heatmap(featuremap[time, :, :], vmax=np.nanmax(featuremap), vmin=np.nanmin(featuremap))
        ax.set_title(times[time])
        plt.show()
        plt.savefig(path, bbox_inches='tight')
        plt.close()

def data_imgs(data, times, savepath, featurename, cmap='Spectral'):
    if not os.path.isdir(savepath):
        os.mkdir(savepath)
    
    fig, ax = plt.subplots()
    for time in range(data.shape[0]):
        path = os.path.join(savepath, f'{featurename}_{time}.png')
        ax = sns.heatmap(data[time, :, :], cmap=cmap, vmax=np.nanmax(data), vmin=np.nanmin(data))
        ax.set_title(times[time])
        plt.savefig(path, bbox_inches='tight')
        plt.close()

def data_gif(path, featurename):
    imgs = []
    for frame in range(len(os.listdir(path))-1):
        imgs.append(imageio.imread(os.path.join(path, f'{featurename}_{frame}.png')))
    imageio.mimsave(os.path.join(path, f'{featurename}.gif'), imgs)

def determine_outlier_thresholds_std(mu, sd, factor):
    """
    Calculates the upper and lower threshold for what is considered an outlier. Currently using 3 standard deviations,
    meaning that 99.7% of the data falls within these boundaries. Alternatively: 68% of the data falls within 1 and
    95% standard deviation of the mean (depends on how radically you want to trim outliers.
    :param mu:
    :param sd:
    :return:
    """
    upper_boundary = mu + factor * sd
    lower_boundary = mu - factor * sd
    return lower_boundary, upper_boundary


def pop_outliers_std(datalist: list, col_name, factor=3):
    lower_boundary, upper_boundary = determine_outlier_thresholds_std(np.mean(datalist), np.std(datalist), factor)
    if np.sum(datalist > upper_boundary) | np.sum(datalist < lower_boundary):
        outliers = list((datalist > upper_boundary) | (datalist < lower_boundary))
        cleaned_df = datalist[[not x for x in outliers]]
        outliers = datalist[outliers]
        return cleaned_df, outliers
    else:
        return datalist, None