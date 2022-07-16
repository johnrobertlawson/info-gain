"""Utility functions.
"""
import pdb

import numpy as np

def bound(x: np.ndarray[int|float]|int|float, lower: int|float, upper: int|float,
            Ne: int, ) -> np.ndarray|int|float:
    if isinstance(x, (int,float)):
        x = np.array(x)
    x[x>upper] = upper
    x[x<lower] = lower
    return x
