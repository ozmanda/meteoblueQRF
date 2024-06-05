import argparse
from DataGenerator import DataGenerator
from FeatureMapGenerator import FeatureMapGenerator
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, help='Mode of the data generation process (training, inference, validation)', default=None)
    parser.add_argument('--datapath', type=str, help='Path to the data directory', default='Data/Messdaten')
    parser.add_argument('--stationinfo', type=str, help='Path to the station information file', default='Data/stations.csv')
    parser.add_argument('--geopath', type=str, help='Path to the geodata directory', default='Data/geodata/')
    parser.add_argument('--savepath', type=str, help='Path to the save directory', default=None)
    parser.add_argument('--palmfile', type=str, help='Path to the PALM simulation file for validation', default=None)
    parser.add_argument('--boundary', type=int, nargs=4, help='Boundary for the inference feature map generation', default=None)
    args = parser.parse_args()
    assert args.mode, 'A mode must be given, choose between "training" and "featuremap"'
    assert args.savepath, 'A savepath must be given'

    if args.mode == 'training':
        if not os.path.isdir(args.savepath):
            os.mkdir(args.savepath)
        datagenerator: DataGenerator = DataGenerator(args.datapath, args.geopath, args.savepath, args.stationinfo)
        datagenerator.generate()

    elif args.mode in ['inference', 'validation']:
        if not os.path.isdir(args.savepath):
            os.mkdir(args.savepath)
        datagenerator: FeatureMapGenerator = FeatureMapGenerator(args.mode, args.datapath, args.stationinfo, args.geopath, args.savepath)
        if args.palmfile: 
            datagenerator.generate(palmfile=args.palmfile)
        elif args.boundary:
            datagenerator.generate(boundary=args.boundary)
        else:
            datagenerator.generate()
        
    else:
        raise ValueError('Invalid mode given, choose between "training" and "featuremap"')

