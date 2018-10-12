import numpy as np


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