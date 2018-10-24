import sys
import xarray as xr
from radiometry import sub_dark


def main(argv):
    inputfile = argv[0]
    darkfile = argv[1]
    outfile = argv[2]

    print(f'Subtracting  (filename coordinate) {darkfile}')
    print(f'from variable dn in {inputfile}')
    print(f'and saving output dataset to {outfile}')
    ok = input('Is this right? [y/n]')
    if ok != 'y':
        print('Aborting...')
        exit()
    
    ds = xr.open_dataset(inputfile, chunks={'filename': 1})
    dark = ds.sel({'filename': darkfile})
    
    print('Data and dark found, subtracting dark')
    ds['dark_corrected_dn'] = sub_dark(ds['dn'], dark['dn'])
    ds = ds.drop('dn')
    ds.assign_attrs({'dark_reference': dark})

    print(f'Finished, saving to {outfile}')
    ds.to_netcdf(outfile)


if __name__ == '__main__':
    main(sys.argv[1:])