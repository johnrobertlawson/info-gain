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

from scipy.stats import entropy
import numpy as N
import matplotlib as M
import matplotlib.pyplot as plt
from numpy import random

def lorenz(x,y,z,sigma=10,b=2.667,r=28):
    xdot = sigma * (y-x)
    ydot = r*x - y - x*z
    zdot = x*y - b*z
    return xdot, ydot, zdot

dt = 0.0005
ntotal=20100
#ntotal=101000
X0=0.5
Y0=1.5
Z0=11.1
r = 166.08
pc = 99.3
nmembers = 50

fig,axes = plt.subplots(nrows=3)

# Fig 1a: time series
ax = axes.flat[0]

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

print("Done.")

Z = Z[1000:]

ax.plot(Z)
pc_y = N.percentile(Z,pc)
ax.axhline(pc_y,color='red',zorder=101,lw=0.5)
ax.set_xlim([0,ntotal-1000])

# Figure 1b: discretisation
ax = axes.flat[1]

exceed = N.where(Z > pc_y,1,0)
exceed = exceed[N.newaxis,:]
ax.pcolormesh(exceed,cmap=M.cm.cubehelix_r)

# Fig 1c: turn into a 30x24 hour "tornado in oklahoma" plot
# 100 days, 24 hours, 41 time steps for each hour
ax = axes.flat[2]
len_array = int((ntotal - 1000)/41)
newarray = N.zeros([len_array])

for i in range(len(newarray)):
    z = Z[i*41:((i+1)*41)+1]
    if z.max() > pc_y:
        newarray[i] = 1
    # pdb.set_trace()
newarray = newarray[N.newaxis,:]
ax.pcolormesh(newarray,cmap=M.cm.cubehelix_r)

fname = "lorenz_test.png"
fpath = fname
fig.tight_layout()
fig.savefig(fpath)

""" For each "truth" time with a True, run two ensembles
with differing "errors" in ICs and random drift/perturbations
to mimic model error. So four places? Do same discretisation. Need to also
create probabilities.

Now evaluate with BS, DS, CSI.

Now allow time tolerance of +/- x hours, and reevaluate

Do we see DS reward rare events better?

Need 50-member ensemble to make sure the pdf is well captured
"""
z_ts = N.zeros([ntotal-999,nmembers])

# Fix random seed
random.seed(1)

for nmem in range(nmembers):
    X = N.empty(ntotal+1)
    Y = N.empty(ntotal+1)
    Z = N.empty(ntotal+1)

    # Add random perturbations to IC-error
    maxpert = 0.0001
    pert = random.uniform(-maxpert,maxpert)

    X[0] = X0 + pert
    Y[0] = Y0 + pert
    Z[0] = Z0 + pert

    # Add random perturbations for model-error
    for n in range(ntotal):
        xdot, ydot, zdot = lorenz(X[n],Y[n],Z[n],r=r)
        X[n+1] = X[n] + (dt*xdot) + (pert*X[n])
        Y[n+1] = Y[n] + (dt*ydot) + (pert*Y[n])
        Z[n+1] = Z[n] + (dt*zdot) + (pert*Z[n])

    print(f"Done ensemble member {nmem+1}.")

    z_ts[:,nmem] = Z[1000:]

# Plot these members
fig,axes = plt.subplots(nrows=6)

ax = axes.flat[0]
ax.plot(Z)
pc_y = N.percentile(Z,pc)
ax.axhline(pc_y,color='red',zorder=101,lw=0.5)
ax.set_xlim([0,ntotal-1000])

for nmem in range(5):
    ax = axes.flat[nmem+1]
    ax.plot(z_ts[:,nmem])
    pc_y = N.percentile(Z,pc)
    ax.axhline(pc_y,color='red',zorder=101,lw=0.5)
    ax.set_xlim([0,ntotal-1000])
    ax.set_ylim([0,300])

fname = "example_ensemble.png"
fpath = fname
fig.tight_layout()
fig.savefig(fpath)

""" More entropy etc
"""

# Entropy of time series
pk = N.sum(newarray)/newarray.size
H = -N.log2(pk)
# This is 6.05373198393779

# A probabilistic entropy is done with KS entropy
# Create probs from the 50 ensemble members


""" Now using real or synthetic data in 2D sequences, with verification,
do decomposition. Can use KSE for H(f) and Shannon entropy for H(o)

Generate fake data from Caers or Korvin?

Or use real WoFS data over HREF.
"""
