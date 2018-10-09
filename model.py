"""
Spectrophotometric model for calibration using color tiles
"""

import numpy as np


def deltaR(R, c1, c2, c3, c4, c5):
    """Difference from the assigned reflectance
    Given model parameters c1-c5 and the reported reflectance R, returns
    the difference given by the model equation from
    ``Spectrophotometry: Accurate Measurement of Optical Properties of Materials´´,
    vol.46, p.394
    """
    dR = spectral_derivative(R)
    ddR = spectral_derivative(dR)

    return _deltaR(R, dR, ddR, c1, c2, c3, c4, c5)


def _deltaR(R, dR, ddR, c1, c2, c3, c4, c5):
    """Difference from the assigned reflectance
    Given model parameters c1-c5 and the reported reflectance R and its derivatives, returns
    the difference given by the model equation from
    ``Spectrophotometry: Accurate Measurement of Optical Properties of Materials´´,
    vol.46, p.394
    """
    return c1 + c2 * R + c3 * dR + c4 * ddR + c5 * (100 - R) * R


def spectral_derivative(x, n=1):
    pass
