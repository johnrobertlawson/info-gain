""" Code for the scale-aware info-gain paper.

* RAW: the raw output of the "truth" that is unclipped for spin-up.
* TRUTH: the clipped raw output that forms the basis for the rest
* TRUTH_ERROR: added decorrelated noise (sine wave?) to mimic obs error
* EVENT_TRUTH: thresholded truth (before/after error?) for events
* EVENT_TRUTH_FILT: 5-step windowing to allow tolerance in time
* GOOD_ENS_n: ensemble members with small errors in ICs and "model error" (perts -
    correlate in time? Must be larger than the obs error or a different nature.)
    - Do n ensemble members, which may require interpolation of probabilities
        when verifying
* BAD_ENS_n: ensemble members with large errors. Similar to above but worse model

Consider doing each of these above for different "regime" by varying the ICs or
    control parameter (r)

Experiments/verification (permutation of some below, e.g. with/without tolerance
        for FSS v IG_sa):
    * Compare time-tolerance vs. no tolerance
    * Compare FSS v IGsa (does this reduce to BS v IGN?)
    * Compare cross-entropy (incl. obs error) vs D_KL
        - This may need checking errors in time v errors in magnitude
        - Check sensitivity to magnitude of observational error
    * Look at decomposed IGsa (and maybe decompose FSS/BS?)
        - Check cross-entropy vs. D_KL again
    * Compare range of regime intermittency (more/less rare)
        - Show spectrum to show long tails - before/after event filtering?
        - Does FSS/BS or IGsa reward rare forecasts better?
    * Compare filtering of events v no filtering (i.e., time windowing performed
        on the continuous data)

Future paths:
    * Do the above on real data - but this may be too much CPU power for right now
    * A self-contained Python script for the experiments
    * A self-contained Python script for real-data verification
        - This might be a future paper
        - Might split into "Scale-aware info gain: verifying idealised intermittency"
            vs. "Scale-aware info gain of gridded weather forecasts"

Requirements/scope:
    * Consider developing modules (lorenz63; verification) and load into master .py

TODO:
    * Everything
"""

import os
import pdb
import multiprocessing

import numpy as N

from lorenz63_old import Lorenz63
import plotting
import verif

# Parallelise throughout!

# CREATE MODEL
L63 = Lorenz63()

# Generate truth with intermittent ICs from that book/paper...
# Constants for truth
r = 166.08
b = 0 # Lorenz 63 default value
sigma = 0 # Lorenz 63 default value

# These three are assigned by Lorenz 1963, rather than for B-R convection 80s papers
X0 = 0
Y0 = 1
Z0 = 0

RAW = L63.generate_time_series(X0=X0,Y0=Y0,Z0=Z0,r=r,b=b,sigma=sigma,)
TRUTH =
