"""
Create "truth" for a time series over a day.

Intermittent peaks of w from Lorenz 63 is like e.g. UH peaking near OKC

Run ensembles from each ICs with a small deviation (error) and drift (random; set constant for each member)
Add a second ensemble with a bigger IC error and model drift
Now verify with time-to-time (average CSI, BS, IG, eFSS). Decompose if possible. 
Do multiple thresholds, getting rarer and rarer.
Now add time tolerance and repeat

We should show 
    (1) IG better rewards rare events, 
    (2) time tolerance greatly increases predict. horizon, and 
    (3) the concept of saturation in this framework (information transfer).

Can also demonstrate estimate of entropy from forecast/observed: for time series, it must be Shannon. 
For spatial fields in time, there's different entropy types...
"""
import os
import pdb

import numpy as N
import matplotlib.pyplot as plt

def lorenz(x,y,z,sigma=10,b=2.667,r=28):
    xdot = sigma * (y-x)
    ydot = r*x - y - x*z
    zdot = x*y - b*z
    return xdot, ydot, zdot

dt = 0.0005
ntotal=100000
X0=0.5
Y0=1.5
Z0=1.55

rs = [166,166.04,166.06,166.08]
fig,axes = plt.subplots(nrows=4)

for ax, r in zip(axes.flat,rs):
    X = N.empty(ntotal+1)
    Y = N.empty(ntotal+1)
    Z = N.empty(ntotal+1)

    X[0] = X0
    Y[0] = Y0
    Z[0] = Z0

    for n in range(ntotal):
        xdot, ydot, zdot = lorenz(X[n],Y[n],Z[n],r=r)
        X[n+1] = X[n] + (dt*xdot) 
        Y[n+1] = Y[n] + (dt*ydot)
        Z[n+1] = Z[n] + (dt*zdot)

    ax.plot(Z)
    print("Done.")

fname = "lorenz_test.png"
fpath = fname
fig.tight_layout()
fig.savefig(fpath)
