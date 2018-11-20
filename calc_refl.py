import xarray as xr
import sys
from dask.diagnostics import ProgressBar
from utils import extract_references
from radiometry import compute_reflectance


ref_coords = {
    'filename': [
        '_I50_L0-511_3-10-2018_10.59.48_White',
        '_I50_L0-511_3-10-2018_13.17.29_White',
        '_I50_L0-511_3-10-2018_13.33.40_White',
        '_I50_L0-511_3-10-2018_13.47.54_PTFE_bottom',
        '_I50_L0-511_3-10-2018_13.50.8_PTFE_',
        '_I50_L0-511_3-10-2018_13.52.45_PTFE_top'
    ]
}


def main(argv):
    inputfile = argv[0]
    output = argv[1]
    variable = argv[2]

    print(f'Calculating reflectances from dataset {inputfile} for')
    print(f'all and saving result to {output}.')
    ds = xr.open_dataset(inputfile, chunks={'filename': 1})
    ds = extract_references(inputfile, variable, ref_coords)
    found_refs = "\n".join(ds["reference_filename"].data)
    print(f'Found references \n{found_refs}')

    ok = input('Is this right? [y/n]')
    if ok != 'y':
        print('Aborting...')
        exit()
    print('Computing reflectances...')
    
    ds = compute_reflectance(ds, variable, f'reference_{variable}')
    with ProgressBar():
        ds.to_netcdf(output)


if __name__ == '__main__':
    main(sys.argv[1:])