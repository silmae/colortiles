import xarray as xr
import utils as u
import sys

def main(argv):
    inputfile = argv[0]
    variable = argv[1]
    output = argv[2]

    outvar = 'mean_' + variable
    confirmation = (f'Calculate spatial mean of variable {variable} in {inputfile}\n'
                    f'and save it as {outvar} in {output}? [y/n]')

    ok = input(confirmation)
    if ok != 'y':
        print('Aborting...')
        exit()

    print(f'Opening dataset')
    ds = xr.open_dataset(inputfile)
    ds['mean' + variable] = ds[variable].mean(dim=['x', 'y'])
    ds = ds.drop(variable)
    ds.to_netcdf(output)

if __name__ == '__main__':
    main(sys.argv[1:])