import xarray as xr
import sys

# White references to calculate reflectance with
reftimes = [
    '2018-10-03T10:59:48.000000000',
    '2018-10-03T13:17:29.000000000',
    '2018-10-03T13:33:40.000000000',
    ]

def main(argv):
    input = argv[0]
    output = argv[1]
    print(f'Calculating reflectances from dataset {input} using refs \n {reftimes}')
    print(f'and saving result to {output}.')
    print(80*'=')
    print('Reading dataset')
    ds = xr.open_dataset(input)
    ds.load()

    refs = ds.sel(time=reftimes)
    refls = []
    print('Calculating reflectances...')
    for t, ref in refs.groupby('time'):
        print(f'Reference {t}')
        refl = ds['dark_corrected_dn'] / ref
        refl.coords['reference'] = t
        refls.append(refl)
    
    print(f'Saving results to {output}')
    ds['reflectance'] = xr.concat(refls, dim='reference')
    ds.drop('dark_corrected_dn')
    ds.to_netcdf(output)



if __name__ == '__main__':
    main(sys.argv[1:])