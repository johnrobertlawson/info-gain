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

# temporal granularity of the integration
dt = 0.0005
#ntotal_raw=20000
ntotal_raw=30000
#ntotal=101000
X0=0.5
Y0=1.5
Z0=11.1
# r = 166.08
r = 166.09
# pc = 99
pc = 97
nmembers = 50
spinup_len = 1000
ntotal = ntotal_raw + spinup_len

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

Z = Z[spinup_len:]
TRUTH = Z

ax.plot(TRUTH)
pc_y = N.percentile(TRUTH,pc)
ax.axhline(pc_y,color='red',zorder=101,lw=0.5)
ax.set_xlim([0,ntotal-spinup_len])

# Figure 1b: discretisation
ax = axes.flat[1]

exceed = N.where(TRUTH > pc_y,1,0)
exceed = exceed[N.newaxis,:]
ax.pcolormesh(exceed,cmap=M.cm.cubehelix_r)

# Fig 1c: turn into a 30x24 hour "tornado in oklahoma" plot
# 100 days, 24 hours, 41 time steps for each hour
ax = axes.flat[2]
len_array = int((ntotal - spinup_len)/41)
newarray = N.zeros([len_array])

# temporal filtering
for i in range(len(newarray)):
    z = Z[i*41:((i+1)*41)+1]
    if z.max() > pc_y:
        newarray[i] = 1
    # pdb.set_trace()
newarray = newarray[N.newaxis,:]
ax.pcolormesh(newarray,cmap=M.cm.cubehelix_r)
TRUTH_FILTERED = newarray

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
z_ts = N.zeros([ntotal-(spinup_len-1),nmembers])



for nmem in range(nmembers):
    X = N.empty(ntotal+1)
    Y = N.empty(ntotal+1)
    Z = N.empty(ntotal+1)

    # Add random perturbations to IC-error
    maxpert = 0.00001
    pert = random.uniform(-maxpert,maxpert)

    X[0] = X0 + pert
    Y[0] = Y0 + pert
    Z[0] = Z0 + pert

    # Add random perturbations for model-error
    for n in range(ntotal):
        xdot, ydot, zdot = lorenz(X[n],Y[n],Z[n],r=r)
        X[n+1] = X[n] + (dt*xdot) + pert #(pert*X[n])
        Y[n+1] = Y[n] + (dt*ydot) + pert #(pert*Y[n])
        Z[n+1] = Z[n] + (dt*zdot) + pert #(pert*Z[n])

    print(f"Done ensemble member {nmem+1}.")

    z_ts[:,nmem] = Z[spinup_len:]


# Plot these members
mems = 4
fig,axes = plt.subplots(nrows=2+(mems*2))

ax = axes.flat[0]
ax.plot(TRUTH)
ax.axhline(pc_y,color='red',zorder=101,lw=0.5)
ax.set_xlim([0,ntotal-spinup_len])
ax.set_ylim([0,300])

ax = axes.flat[1]
ax.pcolormesh(newarray,cmap=M.cm.cubehelix_r)

members_plot = N.arange(mems)
axes_plot = (N.arange(mems)*2)+2

for nmem, nax in zip(members_plot,axes_plot):
    ax = axes.flat[nax]
    ax.plot(z_ts[:,nmem])
    pc_y = N.percentile(Z,pc)
    ax.axhline(pc_y,color='red',zorder=101,lw=0.5)
    ax.set_xlim([0,ntotal-spinup_len])
    ax.set_ylim([0,300])

    newarray = N.zeros([len_array])

    ax = axes.flat[nax+1]

    # temporal filtering
    for i in range(len(newarray)):
        z = z_ts[:,nmem][i*41:((i+1)*41)+1]
        if z.max() > pc_y:
            newarray[i] = 1
        # pdb.set_trace()
    newarray = newarray[N.newaxis,:]
    ax.pcolormesh(newarray,cmap=M.cm.cubehelix_r)
    print("Plotted axis",nax+1)

fname = "example_ensemble.png"
fpath = fname
fig.tight_layout()
fig.savefig(fpath)


# Plot JUST FAKE NADER DAYS
nmems = 9
fig,axes = plt.subplots(nrows=nmems+1,figsize=(15,12))

ax = axes.flat[0]
ax.pcolormesh(newarray,cmap=M.cm.Reds)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)

members_plot = N.arange(nmems)
axs = axes.flat[1:]
for nmem, nax in zip(members_plot,axs):
    # ax = axes.flat[nax]
    ax = nax
    pc_y = N.percentile(Z,pc)
    # ax.set_xlim([0,ntotal-spinup_len])
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    # ax.set_ylim([0,300])

    newarray = N.zeros([len_array])

    # temporal filtering
    for i in range(len(newarray)):
        z = z_ts[:,nmem][i*41:((i+1)*41)+1]
        if z.max() > pc_y:
            newarray[i] = 1
    newarray = newarray[N.newaxis,:]
    ax.pcolormesh(newarray,cmap=M.cm.cubehelix_r)
    print("Plotted axis",nmem)

fname = "firstgroup_ensemble_discrete.png"
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
