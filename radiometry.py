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


def direction(destination, origin): 
    """Get the unit direction vector from a point to another.
    
    Parameters
    ----------
    destination : array-like
        1 x 3 array of coordinates of the destination point.
    origin : array-like
        N x 3 array of coordinates of the origin point(s).

    Returns
    -------
    np.ndarray
        N x 3 array of unit vectors from the origin(s) to destination.
    """
    res = np.asarray(destination) - np.asarray(origin)
    return res / np.linalg.norm(res, axis=1).reshape(-1, 1)


def cosine_for(light, points):
    """Calculate the cosine correction given points on the surface
    ([x, y, 0]) and the point light source at [x', y', z'].

    Parameters
    ----------
    light : array-like
        1 x 3 array of coordinates of the point light source.
    points : array-like
        N x 2 array of coordinates of the surface point(s) at z=0.
    
    Returns
    -------
    result : array-like
        N x 1 Array of cosines for the given surface points.
    """
    directions = direction(
        light, 
        np.hstack([points, np.zeros((points.shape[0], 1))])
        )
    normal = np.array([0, 0, 1])
    return np.dot(normal, directions.T).reshape(-1, 1)