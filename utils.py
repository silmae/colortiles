"""
Utility functions.
"""
import xarray as xr
import os.path as osp
from datetime import datetime
from math import ceil


def band2wl(x):
    """Swap band to wavelength"""
    return x.swap_dims({'band': 'wavelength'})


def extract_references(ds, variable, ref_coords):
    """Extract reference data to a new variable in the dataset.
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the given variable and references.
    variable : str
        Variable to extract from the dataset.
    ref_coords : dict
        Coordinates of each reference as {'dim': [values]}
    
    Returns
    -------
    xr.Dataset
        Dataset containing the references in a new variable
        `reference_variable` with the given reference coordinates
        as `reference_dim` for each dimension used.
    """

    refs = ds.sel(**ref_coords)[variable]
    for k in ref_coords:
        refs.coords[f'reference_{k}'] = refs.coords[k]
        refs = refs.swap_dims({k: f'reference_{k}'}).drop(k)
    ds[f'reference_{variable}'] = refs
    return ds


def read_ENVI_data(files, variable, **kwargs):

    def process_single_ENVI(file, **kwargs):
        with xr.open_rasterio(file, **kwargs) as da:
            da.coords['filename'] = simple_name(file)
            da.load()
            return da

    data = [process_single_ENVI(f) for f in files]

    ds = xr.Dataset(
        data_vars={variable: xr.concat(data, dim='filename')}
        )

    ds = ds.reset_coords()
    return ds


def simple_name(path):
    """Extract more relevant part of the filename"""
    return osp.splitext(osp.basename(path))[0]


def parse_time(s):
    """Parse the time format of the camera-given filenames to datetime."""
    return datetime.strptime(s, '%d-%m-%Y_%H.%M.%S')


def crop_center(data, n):
    """Select a n x n -sized square in the center of the image.
    For odd width or height, bias towards index 0.
    """
    cx, cy = data.sizes['x'] // 2, data.sizes['y'] // 2
    start, end = n // 2, ceil(n / 2)
    crop = {
        'x': slice(cx - start, cx + end),
        'y': slice(cy - start, cy + end)
         }
    return data.isel(**crop)


def standard_color(s):
    """Standardize the spelling of a given color/BCRA tile
    
    Equates 'Bright yellow' and variants as 'Yellow' 
    """
    replacements = {
        'bright yellow': 'yellow'
    }

    res = s.lower()

    if res in replacements:
        res = replacements[res]
    
    return res


def spatial_points(ds):
    """Gather the coordinates of each spatial point x, y in the dataset.
    
    Assumes that both coordinates have the same dtype.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing coordinates x and y.
    
    Returns
    -------
    np.ndarray
        Array containing x and y coordinates of each point in the dataset.
    """
    if ds.x.dtype != ds.y.dtype:
        raise TypeError(
            (f'x and y coordinates do not have the same dtype:'
             f'x: {ds.x.dtype} != y: {ds.y.dtype}'))

    pts = ds.stack(
        {'xy': ['x', 'y']}
        ).xy.data.astype(f'{ds.x.dtype}, {ds.x.dtype}')
    return pts.view(f'{ds.x.dtype}').reshape(pts.shape + (-1,))
    