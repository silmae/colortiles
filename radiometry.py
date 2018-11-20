import numpy as np
import xarray as xr
from scipy.stats import linregress


def linear_regression(ds, x, y, dim):
    """Compute a linear regression between two variables
    grouped by given dimension.
    
    Parameters
    ----------
    ds : xr.Dataset
      Dataset containing variables x and y with dimension dim.
    x : str
      Independent variable in the dataset.
    y : str
      Dependent variable in the dataset.
    dim : str
      Dimension to group the data by before the regression.
    
    Returns
    -------
    xr.Dataset
      Dataset with the results of the regression along the given dimension.
    """

    def fit_band(ds):
        x_ = ds[x].data.ravel()
        y_ = ds[y].data.ravel()
        res = linregress(x_, y_)
        (ds['slope'],
         ds['intercept'],
         ds['rvalue'],
         ds['pvalue'],
         ds['stderr']) = res
        return ds
    
    ds = ds.groupby(dim).apply(fit_band)

    return ds


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


def compute_reflectance(ds, data_variable, reference_variable):
    """Compute reflectance given data and reference.
  
    Drops both used variables from the resulting dataset, keeping others.
    
    Parameters
    ----------
    ds : xr.Dataset
      Dataset containing both the data and reference variables.
    data_variable : str
      Variable containing the data.
    reference_variable : str
      Variable containing the reference data.
    
    Returns
    -------
    xr.Dataset
      Dataset with the computed reflectance.
    """

    ds['reflectance'] = ds[data_variable] / ds[reference_variable]
    ds = ds.drop([data_variable, reference_variable])
    return ds


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