"""
Utility functions.
"""
import xarray as xr
from glob import glob
import os.path as osp

def band2wl(x):
    """Swap band to wavelength"""
    return x.swap_dims({'band':'wavelength'})


def read_ENVI_ds(pattern):
    files = sorted(glob(pattern))
    return xr.Dataset(
        data_vars=dict((simple_name(f), xr.open_rasterio(f)) for f in files)
    )

def simple_name(path):
    """Extract more relevant part of the path"""
    return osp.basename(path)[22:]