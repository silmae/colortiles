import sys
import xarray as xr
from glob import glob
from radiometry import sub_dark
from utils import read_ENVI_data

from warnings import filterwarnings
filterwarnings('ignore', 'Dataset has no geotransform set')


def main(argv):
    pattern = argv[0]
    darkfile = argv[1]
    outfile = argv[2]

    inputs = sorted(glob(pattern))

    print(f'Subtracting {darkfile} from the following files')
    print(f'and saving output dataset to {outfile}')
    print(80 * '=')
    for i in inputs:
        print(i)
    print(80 * '=')
    ok =input('Is this right? [y/n]')
    if ok != 'y':
        print('Aborting...')
        exit()
    dark = xr.open_rasterio(darkfile)
    print('Dark read, reading dataset')
    ds = read_ENVI_data(inputs)
    ds.load()
    print('Data read, subtracting dark')
    ds = ds.set_coords('wavelength')
    ds['dark_corrected_dn'] = sub_dark(ds['dn'], dark)
    ds = ds.drop('dn')
    ds = ds.reset_coords()

    print(f'Finished, saving to {outfile}')
    ds.to_netcdf(outfile)
    

if __name__ == '__main__':
    main(sys.argv[1:])