"""
Spectrophotometric model for calibration using color tiles
"""

import numpy as np
import xarray as xr
from scipy.linalg import lstsq


def fit_dataarray(R, R_ref):
    """Fit the model parameters to the given measurements R and R_ref.
    
    Parameters
    ----------
    R : xr.DataArray
        Measured reflectance spectra as a DataArray containing wavelength
        coordinates.
    R_ref : xr.DataArray
        Reflectance spectra of the reference with matching wavelength
        coordinates.
    
    """

    ds = xr.Dataset(
        data_vars={
            'deltaR': R_ref - R,
            'R': R,
            'dR': spectral_derivative(R, 1),
            'ddR': spectral_derivative(R, 2)
        }
    )

    A = _deltaR_matrix(
        ds['R'].values.ravel(),
        ds['dR'].values.ravel(),
        ds['ddR'].values.ravel(),
    )

    return lstsq(A, ds['deltaR'].values.ravel())


def _deltaR_matrix(R, dR, ddR):
    """Matrix corresponding to the function _deltaR
         
    Parametersq
    ----------
    R : np.array
        N x 1 vector of reflectance values
    dR : np.array
        N x 1 vector of differentiated reflectance values
    ddR : np.array
        N x 1 vector of twice differentiated reflectance values

    Result
    ------
    np.array
        Matrix for calculating the model

    """
    return np.stack([
        np.ones_like(R),
        R,
        dR,
        ddR,
        (100 - R) * R
        ]
    ).T


def deltaR(R, c1, c2, c3, c4, c5):
    """Difference from the assigned reflectance
    Given model parameters c1-c5 and the reported reflectance R, returns
    the difference given by the model equation from
    ``Spectrophotometry: Accurate Measurement of Optical Properties
    of Materials´´,
    vol.46, p.394
    """
    dR = spectral_derivative(R, n=1)
    ddR = spectral_derivative(R, n=2)

    return _deltaR(R, dR, ddR, c1, c2, c3, c4, c5)


def _deltaR(R, dR, ddR, c1, c2, c3, c4, c5):
    """Difference from the assigned reflectance
    Given model parameters c1-c5 and the reported reflectance R and its
    derivatives, returns the difference given by the model equation from
    ``Spectrophotometry: Accurate Measurement of Optical Properties
    of Materials´´,
    vol.46, p.394
    """
    return c1 + c2 * R + c3 * dR + c4 * ddR + c5 * (100 - R) * R


def spectral_derivative(x, n=1):
    """Calculate the first or second spectral derivatives"""
    return x.differentiate('wavelength', n)