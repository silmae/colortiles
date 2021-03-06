import xarray as xr
import numpy as np
import utils as u
import sys


def main(argv):
    inputfile = argv[0]
    variable = argv[1]
    n = int(argv[2])
    output = argv[3]

    outvar = 'mean_' + variable
    confirmation = (f'Calculate spatial mean of {n} x {n} center pixels of '
                    f'variable {variable} in {inputfile} and save it '
                    f'as {outvar} in {output}? [y/n] ')

    ok = input(confirmation)
    if ok != 'y':
        print('Aborting...')
        exit()

    print(f'Opening dataset')
    ds = xr.open_dataset(inputfile)
    center = u.crop_center(ds[variable], n)
    print('Storing crop mask')
    nonspatialcoords = [c for c in ds[variable].coords if c not in ['x', 'y']]
    ds['cropped_area'] = xr.ones_like(
        center.isel({c: 0 for c in nonspatialcoords}).drop(nonspatialcoords),
        dtype=np.bool8
        )
    print(f'Calculating mean and std for variable {variable}')
    ds['mean_' + variable] = center.mean(dim=['x', 'y'])
    ds['std_' + variable] = center.std(dim=['x', 'y'])
    ds = ds.drop(variable)
    print(f'Saving result to {output}')
    ds.to_netcdf(output)


if __name__ == '__main__':
    main(sys.argv[1:])