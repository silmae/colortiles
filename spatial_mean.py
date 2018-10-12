import xarray as xr
import utils as u
import sys


def main(argv):
    inputfile = argv[0]
    variable = argv[1]
    n = argv[2]
    output = argv[3]

    outvar = 'mean_' + variable
    confirmation = (f'Calculate spatial mean of n x n center pixels of '
                    f'variable {variable} in {inputfile}\n and save it '
                    f'as {outvar} in {output}? [y/n]')

    ok = input(confirmation)
    if ok != 'y':
        print('Aborting...')
        exit()

    print(f'Opening dataset')
    ds = xr.open_dataset(inputfile)
    center = u.crop_center(ds[variable], n)
    ds['cropx'] = center.x
    ds['cropy'] = center.y
    ds['mean_' + variable] = center.mean(dim=['x', 'y'])
    ds['var_' + variable] = center.var(dim=['x', 'y'])
    ds = ds.drop(variable)
    ds.to_netcdf(output)


if __name__ == '__main__':
    main(sys.argv[1:])