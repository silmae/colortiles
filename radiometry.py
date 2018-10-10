import xarray as xr
import numpy as np


def sub_dark(arr, dark, method='zeroclip'):
    """Subtract dark from arr"""
    methods = {
		'zeroclip': _subclip,
		'asfloat': _floatsub,
		}
    def sub(x):
      return methods[method](x, dark)
    
    return xr.apply_ufunc(sub, arr)


def _subclip(x, y):
    return (x > y) * (x - y)


def _floatsub(x, y):
    return np.float64(x) - np.float64(y)