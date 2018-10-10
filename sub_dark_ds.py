import sys
import xarray as xr
from glob import glob
from radiometry import sub_dark
from utils import read_ENVI_ds

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

    dark = xr.open_rasterio(darkfile)
    print('Dark read')
    ds = read_ENVI_ds(inputs)
    print(ds)
    print('Data read, subtracting dark')
    ds = ds.apply(sub_dark, args=(dark,))
    print(f'Finished, saving to {outfile}')
    ds.to_netcdf(outfile)
    

if __name__ == '__main__':
    main(sys.argv[1:])