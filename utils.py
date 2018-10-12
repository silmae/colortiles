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


def read_ENVI_data(files):
    data = dict((simple_name(f), xr.open_rasterio(f)) for f in files)

    for s, da in data.items():
        sparts = s.split('_')
        da.coords['time'] = parse_time('_'.join(sparts[3:5]))
        da.coords['material'] = ' '.join(sparts[5:])

    ds = xr.Dataset(
        data_vars={'dn': xr.concat(data.values(), dim='time')}
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