"""
Utility functions.
"""


def band2wl(x):
    """Swap band to wavelength"""
    return x.swap_dims({'band':'wavelength'})