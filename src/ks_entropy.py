"""
TODO:
    * Expand or put into different branch

"""
import os
import pdb

import numpy as N

def H_ks():
    data = N.zeros()
    for nr,r in N.enumerate(routes):
        for na,a in N.enumerate(atoms):
            PI = N.multiply(probs[nr,na,:])
            data[nr,na] = 

