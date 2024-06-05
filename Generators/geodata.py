import os
import numpy as np
import rasterio
import pickle
import datautils
from warnings import warn
import pandas as pd
from spatialconvolutions import convolutions
import pvlib.irradiance as rad
from typing import List

#! fixed to be meaningful for a resolution of 16m, original convs are [10, 30, 100, 200, 500]
#* convs must always be even when divided by the resolution (considers either side)
convs = [32, 48, 112, 208, 512]

def geogen(geopath: str, boundary: dict, humimaps: np.ndarray):
    geofeaturelist = ["altitude", "buildings", "forests", "pavedsurfaces", "surfacewater", "urbangreen"]
    geomaps = generate_geomap(geopath, boundary, humimaps.shape, geofeaturelist, convs)
    return geomaps

#! This now returns a dictionary, but is untested.
def generate_geomap(geopath: str, boundary: dict, shape: tuple, geofeaturelist: list, 
                    convs: list, sigma:int = 3, resolution=16):
    # TODO: change to dict (done but untested!)
    '''
    Generates a geomap from the given geofeatures and convolutions.
    :param geopath: path to geofeatures
    :param boundary: boundary of the map
    :param shape: shape of the map, already reduced to the resolution of the PALM simulation files
    :param geofeaturelist: list of geofeatures
    :param convs: list of convolutions
    :param sigma: sigma for gaussian convolution
    :return: geomap with shape (n_geofeatures * n_convolutions+1, height of humimap, width of humimap)
    '''
    geomaps = {}
    # geomaps is in the shape which considers the resolution (i.e., it is already reduced)
    print('\nGenerating Geofeatures')
    for idx, geofeature in enumerate(geofeaturelist):
        print(f'    {geofeature}...')
        # load feature map, get border and check that it is complete. Any negative values are assign NaN
        featuremap, geo_border = load_geomap(os.path.join(geopath, f'{geofeature}.tif'))
        featuremap[featuremap < 0] = np.nan
        if boundary['CH_E'] > geo_border['E'] or boundary['CH_W'] < geo_border['W'] or \
                boundary['CH_S'] < geo_border['S'] or boundary['CH_N'] > geo_border['N']:
            warn(f'geofeature map {geofeature} incomplete', Warning)
            print(f'Border geomap:     N {int(geo_border["N"])}, S {int(geo_border["S"])}, W {int(geo_border["W"])}, E {int(geo_border["E"])}')
            print(f'Border featuremap: N {int(boundary["CH_N"])}, S {int(boundary["CH_S"])}, W {int(boundary["CH_W"])}, E {int(boundary["CH_E"])}')
            raise ValueError

        # indices locating the PALM simulation map in the geofeature map
        palm_geoidxs = {'N': int(np.round(geo_border['N'] - boundary['CH_N'])),
                        'S': int(np.round(geo_border['N'] - boundary['CH_S'])),
                        'W': int(np.round(boundary['CH_W'] - geo_border['W'])),
                        'E': int(np.round(boundary['CH_E'] - geo_border['W']))}
        
        # add uncovoluted feature map to geomaps after adjusting to the proper resolution
        geomaps[geofeature] = datautils.reduce_resolution(featuremap[palm_geoidxs['N']:palm_geoidxs['S'], 
                                                                     palm_geoidxs['W']:palm_geoidxs['E']],
                                                                     resolution)        
        if geofeature != 'altitude':
            # create padded feature map for convolutions - full resolution
            padded_featuremap = np.empty(shape=((shape[1]*resolution + np.max(convs)), 
                                                (shape[2]*resolution + np.max(convs))))
            padded_featuremap[:] = np.NaN

            # calculate the amount that padding exceeds geofeature map per edge
            #* negative values indicate that padding exceeds the map in that direction
            padding_over = {'N': int(palm_geoidxs['N'] - (np.max(convs)/2)),
                    'S': featuremap.shape[0] - int(palm_geoidxs['S'] + (np.max(convs)/2)),
                    'W': int(palm_geoidxs['W'] - (np.max(convs)/2)),
                    'E': featuremap.shape[1] - int(palm_geoidxs['E'] + (np.max(convs)/2))}

            for key in padding_over.keys():
                if padding_over[key] < 0:
                    padding_over[key] = abs(padding_over[key])
                else: 
                    padding_over[key] = 0

            # two index sets: 1. start and end of padded featuremap covered by the geomap (assumption is the whole padded map)
            #                 2. indices for the start and end of the padded featuremap within the geomap (used to extract data)
            filled_paddedmap = {'N': 0,
                                'S': padded_featuremap.shape[0]-1,
                                'W': 0,
                                'E': padded_featuremap.shape[1]-1}
            geomapidxs_padded = {'N': int(palm_geoidxs['N'] - (np.max(convs)/2)),
                                'S': int(palm_geoidxs['S'] + (np.max(convs)/2) -1),
                                'W': int(palm_geoidxs['W'] - (np.max(convs)/2)),
                                'E': int(palm_geoidxs['E'] + (np.max(convs)/2) -1)}
            if padding_over['N']:
                geomapidxs_padded['N'] = 0
                filled_paddedmap['N'] = padding_over['N']
            if padding_over['S']:
                geomapidxs_padded['S'] = featuremap.shape[0]
                filled_paddedmap['S'] = padded_featuremap.shape[0] - padding_over['S']
            if padding_over['W']:
                geomapidxs_padded['W'] = 0
                filled_paddedmap['W'] = padding_over['W']
            if padding_over['E']:
                geomapidxs_padded['E'] = 0
                filled_paddedmap['E'] = padded_featuremap.shape[1] - padding_over['E']

            # fill geodata into padded map using the indices
            padded_featuremap[filled_paddedmap['N']:filled_paddedmap['S']+1, 
                            filled_paddedmap['W']:filled_paddedmap['E']+1] = featuremap[geomapidxs_padded['N']:geomapidxs_padded['S']+1,
                                                                                        geomapidxs_padded['W']:geomapidxs_padded['E']+1]

            # reduce padded_featuremap resolution
            padded_featuremap = datautils.reduce_resolution(padded_featuremap, resolution=resolution)
            # padded_featuremap[padded_featuremap == 0] = np.nan
            # add convoluted feature maps to geomaps
            print('    convolutions...')
            max_conv_pad = (np.max(convs)/resolution)/2
            array_idxs = {'N': int(0+max_conv_pad), 
                        'S': int(padded_featuremap.shape[0]-max_conv_pad),
                        'W': int(0+max_conv_pad), 
                        'E': int(padded_featuremap.shape[1]-max_conv_pad)}
            for conv in convs:
                convname = f'{geofeature}_{conv}'
                conv_pad = (conv/2)/resolution
                print(f'      conv {conv}')
                # empty array for convolutions in reduced size
                conv_array = np.zeros(shape=(shape[1], shape[2]))
                kernel = signal.gaussian(conv/resolution + 1, std=sigma) # type: ignore
                kernel = np.outer(kernel, kernel)

                # fill by column in row
                
                for lat in range(0, conv_array.shape[0]):
                    for lon in range(0, conv_array.shape[1]): 
                        geo_array = padded_featuremap[int(lat+array_idxs['N']-conv_pad): int(lat+array_idxs['N']+conv_pad)+1,
                                                    int(lon+array_idxs['W']-conv_pad): int(lon+array_idxs['W']+conv_pad)+1]
                        if geo_array.shape != kernel.shape:
                            print(geo_array.shape)
                            print('padded feature map excerpt:')
                            print(f'    {int(lat+array_idxs["N"]-conv_pad)}:{int(lat+array_idxs["N"]+conv_pad)+1}')
                            print(f'    {int(lon+array_idxs["W"]-conv_pad)}:{int(lon+array_idxs["W"]+conv_pad)+1}')
                            print('Cell:')
                            print(f'    lat: {lat}/{conv_array.shape[0]}\n    lon: {lon}/{conv_array.shape[1]}')
                        gaussian_array = geo_array * kernel
                        conv_array[lat, lon] = np.nanmean(gaussian_array)
                if np.sum(np.isnan(conv_array)) != 0:
                    print(np.sum(np.isnan(conv_array)))
                geomaps[convname] = conv_array
    return geomaps


def load_geomap(path: str):
    featuremap = rasterio.open(path)
    geo_N = featuremap.meta['transform'][5]  # gives northern boundary
    geo_W = featuremap.meta['transform'][2]  # gives western boundary
    # transform to LV95 coordinates
    geo_N, geo_W = datautils.lv03_to_lv95(geo_N, geo_W)
    geo_S = geo_N - featuremap.shape[1]
    geo_E = geo_W + featuremap.shape[0]
    featuremap = featuremap.read()[0, :, :]
    featuremap[featuremap < 0] = 0
    borders = {'N': geo_N, 'S': geo_S, 'W': geo_W, 'E': geo_E}
    return featuremap, borders


def get_loc(stationid: str, infofile: pd.DataFrame):
    """
    Determines the latitude and longitude of the measurement station using
    the input parameter stationid: the ID of the station, corresponding to
    stationid_new in the stations.csv list.

    Returns the latitude and longitude of the station corrected to LV09
    coordinates
    """
    targetlat = None
    targetlon = None
    for _, row in infofile.iterrows():
        if row['stationid_new'] == stationid:
            targetlat = row['CH_N'] if row['CH_N'] < 1000000 else row['CH_N'] - 1000000
            targetlon = row['CH_E'] if row['CH_E'] < 2000000 else row['CH_E'] - 2000000
            break
    if not targetlat and not targetlon:
        warn(f'The location of the sensor for station with ID {stationid} could not be determined', Warning)
        raise ValueError

    return targetlat, targetlon


def extract_feature(targetlat: int, targetlon: int, featurename: str, geopath: str, num: int) -> dict:
    if f'{featurename}.tif' in os.listdir(geopath):
        featuremap = rasterio.open(os.path.join(geopath, f'{featurename}.tif'))
        originlat = featuremap.meta['transform'][5]  # gives northern boundary
        originlon = featuremap.meta['transform'][2]  # gives western boundary
        featuremap = featuremap.read()
        featuremap[featuremap == -9999] = 0

        if targetlat > originlat or targetlon < originlon:
            warn(f'station outside of feature map area for feature {featurename}. \nLatitude: {targetlat} \n'
                 f'Longitude: {targetlon}', Warning)
            raise ValueError

        idxlat = originlat - targetlat
        idxlon = targetlon - originlon

        if idxlat > featuremap.shape[1] or idxlon > featuremap.shape[2]:
            warn(f'station outside of feature map area for feature {featurename}. \nLatitude: {targetlat} \n'
                 f'Longitude: {targetlon}', Warning)
            raise ValueError

    else:
        warn(f'The .tif file for the feature {featurename} is not available in the folder {geopath}, '
             f'check path', Warning)
        raise FileNotFoundError

    if featurename == 'altitude':
        featurevector = [featuremap[0, int(idxlat), int(idxlon)]]*num
        return {featurename: featurevector}
    else:
        featuredict = {featurename: [featuremap[0, int(idxlat), int(idxlon)]]*num}
        featurenames = [f'{featurename}_{conv}' for conv in convs]
        featureconvs = convolutions(convs, featuremap, idxlat, idxlon)
        for idx, feature in enumerate(featurenames):
            featuredict.update({feature: [featureconvs[idx]]*num})
        return featuredict


def get_geofeatures(stationid: str, geopath: str, num: int, geofeaturelist: List[str], infofile: str):
    # list of static / geographic features to be considered
    geofeatures = np.empty(shape=(num, 0))
    targetlat, targetlon = get_loc(stationid, infofile)

    for feature in geofeaturelist:
        geofeatures = np.append(geofeatures, extract_feature(targetlat, targetlon, feature, geopath, num), axis=1)

    return geofeatures, targetlat, targetlon


def load_geofeatures(geodir: str, shape: tuple):
    """
    This function either loads the geofeatures from an existing .PICKLE file, or it generate the
    feature data using the shape of the new NetCDF file and its layermasks.

    PARAMETERS:
    -----------
    topodir: topodir is the directory in which the topographical / geographical data can be found
      By default, it should be located in the (wd)/Data/geofeatures

    shape: this variable is the shape of the original NetCDF file, used to reformat the rasters
      to the dimensions of the training data: (time, layer, height, width)


    RETURNS:
    --------
    geofeatures: this variable contains the features of the 6 geographical features buildings,
      paved surfaces, forests, surface water, urban green and altitude. Geofeatures already have
      invalid observations removed according to the parameter layermasks.
      Attention: these values are hard-coded, as are as their convolutions.

    geolist: contians the names of the geographical variables for which features were loaded or
      generated (appended to the featurelist in NetCDF.py)

    """
    # altitude must always be the last geofeature to be processed
    geolist = ["buildings", "pavedsurfaces", "forests", "surfacewater", "urbangreen", "altitude"]
    rasterdim = (shape[2], shape[3])  # dimensions for the raster/map: layers x height x width

    # standardisted filename
    filename = f'geofeatures_{len(geolist)}_{shape[0]}x{shape[1]}x{shape[2]}x{shape[3]}.PICKLE'

    # check if the geofeatures file already exists and load data if it does, otherwise generate the array
    if os.path.isfile(os.path.join(geodir, filename)):
        geofeatures = pickle.load(open(os.path.join(geodir, filename), "rb"))  # without mask removed
    else:
        # create empty data array with 5 dimensions: times x layers x height x width x geofeatures
        geofeatures = np.zeros(shape=(shape[0], shape[1], shape[2], shape[3], len(geolist)))

        for idx, dataname in enumerate(geolist[0:-1]):
            # load either from .TIF or from .PICKLE
            if os.path.isfile(os.path.join(geodir, f'{dataname}.PICKLE')):
                data = pickle.load(open(os.path.join(geodir, f'{dataname}.PICKLE'), "rb"))
            elif os.path.isfile(os.path.join(geodir, f'{dataname}.tif')):
                data = rasterio.open(os.path.join(geodir, f'{dataname}.tif'))
                data = data.read(1)
                pickle.dump(data, open(os.path.join(geodir, f'{dataname}.PICKLE'), "wb"))  # save geofeatures as PICKLE
            else:
                print(f'Data for geofeature {dataname} is not available in .PICKLE or .tif format')
                break

            if dataname != "altitude":
                # convert the raster into the same dimensions as the training data (rasterdim)
                data = convert_raster(data, rasterdim)
                for time in range(shape[0]):
                    for layer in range(shape[1]):
                        geofeatures[time, layer, :, :, idx] = data

            else:
                # correct the altitude for each layer
                for layer in range(shape[1]):
                    layerdat = data + np.full(shape=data.shape, fill_value=layer*4)
                    for time in range(shape[0]):
                        geofeatures[time, layer, :, :, idx] = layerdat

    return geofeatures, geolist


def convert_raster(data: np.ndarray, newdim: tuple):
    newraster = np.zeros(shape=newdim)

    istep = np.floor(data.shape[0] / newdim[0])
    jstep = np.floor(data.shape[1] / newdim[1])

    for i in range(0, newdim[0]):
        for j in range(0, newdim[1]):
            newraster[i, j] = np.mean(data[int(i*istep):int(i*istep+istep+1), int(j*jstep):int(j*jstep+jstep+1)])

    return newraster

