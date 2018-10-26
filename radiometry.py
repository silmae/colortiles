import numpy as np
import xarray as xr


def PTFE_reflectance_factor_45_0():
    """Reflectance factor as a function of wavelength for 45° incident angle
    and 0° viewing angle of the pressed PTFE plaque as given in
    ````
    """

    wls = np.arange(380, 781, 5, dtype=np.int64)
    R = np.concatenate((
      1.011 * np.ones((8,)),
      1.012 * np.ones((18,)),
      1.015 * np.ones((55,))
    ))
    Rf = xr.DataArray(
      R,
      dims='wavelength',
      coords={'wavelength': wls}
    )
    return Rf


def sub_dark(arr, dark, method='default'):
    """Subtract dark from arr, clipping to 0"""
    methods = {
      'default': _subclip,
      'ymean': lambda a, b: _subclip(b.mean(dim='y')),
      }

    return methods[method](arr, dark)


def _subclip(x, y):
    return (x > y) * (x - y)


def _floatsub(x, y):
    return np.float64(x) - np.float64(y)