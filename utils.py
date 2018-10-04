"""
Utility functions.
"""

import numpy as np

def sub_dark(arr, dark, method='direct'):
    """Subtract dark from arr"""
    methods = {
            'direct': subclip,
            'xmean': lambda x,y: subclip(x.groupby('x'), y.mean(dim='x'))
            }
    return methods[method](arr, dark)

def subclip(x,y):
    return (x > y) * (x - y)
