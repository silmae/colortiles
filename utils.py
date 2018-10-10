"""
Utility functions.
"""
import xarray as xr
from glob import glob
import os.path as osp
from datetime import datetime


def band2wl(x):
    """Swap band to wavelength"""
    return x.swap_dims({'band':'wavelength'})


def read_ENVI_ds(files):
    data = dict((simple_name(f), xr.open_rasterio(f)) for f in files)
    
    for s, da in data.items():
        sparts = s.split('_')
        da.coords['time'] = parse_time('_'.join(sparts[3:5]))
        da.coords['material'] = ' '.join(sparts[5:])
    
    return xr.concat(data.values(), dim='time')


def simple_name(path):
    """Extract more relevant part of the filename"""
    return osp.splitext(osp.basename(path))[0]


def parse_time(s):
    return datetime.strptime(s, '%d-%m-%Y_%H.%M.%S')