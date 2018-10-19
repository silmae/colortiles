import xarray as xr
import sys

# Distance coordinate (follows order of the following indices)
distances = {
    'distance': ('time', ['near', 'middle', 'far', 'middle', 'near', 'far'])
    }
# indices of the green tiles at different distances
green_idx = [12, 17, 21]
# indices of the PTFEs at different distances
ptfe_idx = [19, 18, 20]


def main(argv):
    inputfile = argv[0]
    outputfile = argv[1]

    print(f'Looking for green tiles and PTFEs from {inputfile}')
    ds = xr.open_dataset(inputfile, chunks={'time': 1})

    print(f'Saving new dataset to {outputfile}')
    new_ds = ds.isel(**{'time': green_idx + ptfe_idx}).assign_coords(**distances)
    new_ds.to_netcdf(outputfile)


if __name__ == '__main__':
    main(sys.argv[1:])