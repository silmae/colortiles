"""Collect a set of ENVI files to a combined NetCDF dataset"""

import sys
import pandas as pd
from glob import glob
from utils import read_ENVI_data

from warnings import filterwarnings
filterwarnings('ignore', 'Dataset has no geotransform set')


def main(argv):
    pattern = argv[0]
    outfile = argv[1]
    variable = argv[2]
    try:
        meta = argv[3]
    except IndexError:
        meta = None

    inputs = sorted(glob(pattern))

    print(f'Collecting the following files to variable {variable}')
    print(f'in file {outfile}:')
    print(80 * '=')
    for i in inputs:
        print(i)
    print(80 * '=')

    if meta is not None:
        df = pd.read_csv(meta)
        df = df.set_index('filename')
        print(f'Including columns {list(df.columns)} from {meta}')
        
    ok = input('Is this right? [y/n]')
    if ok != 'y':
        print('Aborting...')
        exit()

    ds = read_ENVI_data(inputs, variable, chunks={})

    if meta is not None:
        ds = ds.assign(df)

    ds = ds.reset_coords()

    print(f'Finished, saving to {outfile}')
    ds.to_netcdf(outfile)


if __name__ == '__main__':
    main(sys.argv[1:])