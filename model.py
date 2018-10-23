"""
Spectrophotometric model for calibration using color tiles
"""

import numpy as np
import xarray as xr
from scipy.linalg import lstsq


def apply_model(R, coefs):
    A = deltaR_matrix(R)
    return R + A.dot(coefs, dims='coefficient')


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
            'A': deltaR_matrix(R)
        }
    )

    def fit(ds):
        res = xr.DataArray(
            lstsq(ds['A'].data.T, ds['deltaR'].data.ravel())[0],
            dims={'coefficient'},
            coords=dict(
                coefficient=['C1', 'C2', 'C3', 'C4', 'C5'],
                ),
            )
        return res

    return ds.groupby('wavelength').apply(fit)


def deltaR_matrix(R):
    dR = spectral_derivative(R, n=1)
    ddR = spectral_derivative(R, n=2)
    return xr.DataArray(
        _deltaR_matrix(R, dR, ddR),
        dims=('coefficient', *R.dims),
        coords=dict(
            coefficient=['C1', 'C2', 'C3', 'C4', 'C5'],
            **R.coords,
            ),
    )


def _deltaR_matrix(R, dR, ddR):
    """Matrix corresponding to the function _deltaR
         
    Parameters
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
    A = np.stack([
        np.ones_like(R),
        R,
        dR,
        ddR,
        (1 - R) * R
        ],
    )
    return A


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
    return c1 + c2 * R + c3 * dR + c4 * ddR + c5 * (1 - R) * R


def _spectral_derivative(x, n=1, dim='wavelength'):
    """Calculate the first or second spectral derivatives"""
    if n == 1:
        res = x.differentiate(dim, 2)
    elif n == 2:
        res = x.differentiate(dim, 2).differentiate(dim, 2)
    else:
        raise(ValueError('Only n=1,2 supported.'))
    
    return res


def spectral_derivative(x, n=1, edge_order=1, dim='wavelength'):
    """Calculate the first or second spectral derivatives
    
    Computes first derivatives with upto second-order accurate edges and second
    derivatives with first order accurate edges.


    Parameters
    ----------
    x : DataArray or Dataset
        Data to be differentiated.
    n : int, optional
        nth derivative, default 1.
    dim : str, optional
        Dimension to differentiate by, default 'wavelength'.
    """

    d = x.coords[dim].diff(dim)
    h = d[0]
    if not np.all(h == d):
        raise(ValueError(f'Coordinate {dim} is not equally spaced.'))

    r = x.rolling(
            {dim: 3}, center=True
        ).construct(
            'window'
        )

    if n == 1:
        Kc = xr.DataArray([-1, 0, 1], dims='window')

        if edge_order == 1:
            Kf = xr.DataArray([0, -1, 1], dims='window') / h
            Kb = xr.DataArray([-1, 1, 0], dims='window') / h
            c_idx = (0, -1)
        elif edge_order == 2:
            Kf = xr.DataArray([-3, 4, -1], dims='window') / (2 * h)
            Kb = xr.DataArray([1, -4, 3], dims='window') / (2 * h)
            c_idx = (1, -2)
        res = r.dot(Kc) / (2 * h)
        
        fd = r.where(~r.isnull(), 0).dot(Kf)
        bd = r.where(~r.isnull(), 0).dot(Kb)

        res[{dim: 0}] = fd[{dim: c_idx[0]}]
        res[{dim: -1}] = bd[{dim: c_idx[1]}]
    elif n == 2:
        K = xr.DataArray([1, -2, 1], dims='window')
        res = r.dot(K) / h**2
    else:
        raise(ValueError('Only n=1,2 supported'))
    
    return res.dropna(dim)