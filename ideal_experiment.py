"""
Generation of time series and verification with multiple scores.

Only do in object-based framework. Here, that is verifying
per observed event (True).

Need to re-generate forecasts initialised from the obs every
x units of "time". For this, take raw ICs from obs, and
perturb slightly (like obs error), and run as normal. This
will generate n forecasts - need to interpolate? Will generate
a lot of forecasts. Need to interpolate with a small enough
granularity to capture the general "tornado day" with
e.g. 4 consequtive time steps. Might need to lower percentile
to make sure it is captured.

Plot stuff on the way to check it is correct.

Entropy can be computed as fraction of time that is True.

FSS ('wrong way'), eFSS, BSS, info gain.
"""


import os
import pdb

from scipy.stats import entropy
from scipy.ndimage import uniform_filter, maximum_filter
import numpy as N
import matplotlib as M
import matplotlib.pyplot as plt
from numpy import random
import skimage

# Fix random seed
random.seed(1)

def lorenz(x,y,z,sigma=10,b=2.667,r=28):
    xdot = sigma * (y-x)
    ydot = r*x - y - x*z
    zdot = x*y - b*z
    return xdot, ydot, zdot

def generate_timeseries(X0,Y0,Z0,ntotal,r,maxpert=None,
                            discretise_size=None,icmulti=1):
    """This is not clipped for spin-up, so
    first 1000 or so points are "off the attractor".
    """
    X = N.empty(ntotal+1)
    Y = N.empty(ntotal+1)
    Z = N.empty(ntotal+1)

    # IC perturbations
    if not maxpert: # For truth
        pert = 0
    else:
        pert = random.uniform(-maxpert,maxpert)

    ICpert = pert * icmulti
    X[0] = X0 + ICpert
    Y[0] = Y0 + ICpert
    Z[0] = Z0 + ICpert

    for n in range(ntotal):
        xdot, ydot, zdot = lorenz(X[n],Y[n],Z[n],r=r)
        # Add model error Here

        if not maxpert: # For truth
            pert = 0
        else:
            pert = random.uniform(-maxpert,maxpert)

        X[n+1] = X[n] + (dt*xdot) + pert
        Y[n+1] = Y[n] + (dt*ydot) + pert
        Z[n+1] = Z[n] + (dt*zdot) + pert

    print("Generated time series.")

    return Z

def discretise_ts(Z,size):
    Z_discrete = N.zeros_like(Z).flatten()
    posts = N.arange(Z_discrete.size-(size+1))[::size]
    for n, pn in enumerate(posts):
        if n != posts.size-1:
            idxn = pn
            idxx = posts[n+1]
            Z_discrete[idxn:idxx] = N.max(Z[0,idxn:idxx])
    print("Discretised time series.")
    return Z_discrete[N.newaxis,:]#.astype(bool)


def identify_features(data,threshold,footprint,dx):
    """ Need to remember to cut times during spin-up time.
    """
    if data.ndim == 1:
        data = data[:,N.newaxis]

    data_masked = N.where(data >= threshold, data, 0)
    data_masked_bool = data_masked > 0.0

    obj_labels = skimage.measure.label(data_masked_bool).astype(int)
    obj_props = skimage.measure.regionprops(obj_labels,data_masked)

    # Now for sifting the objects based on criteria
    obj_OK = []
    for prop in obj_props:
        label = int(prop.label)

        # Object must be bigger than (footprint/dx**ndim)
        size_OK = False

        if prop.area > footprint:
            size_OK = True

        if size_OK: # and... etc
            obj_OK.append(label)

    data_objs = N.zeros_like(data_masked)
    # This replaces gridpoints within objects with original data
    for i,ol in enumerate(obj_OK):
         data_objs = N.where(obj_labels==ol, data_masked, data_objs)
    #    obj_labels_OK =  N.where(obj_labels==ol, obj_labels, 0)
#        obj_props_OK = [obj_props[o-1] for o in obj_OK]
    # pdb.set_trace()
    return

# temporal granularity of the integration
dt = 0.0008

# Desired number of integration steps
ntotal_clipped = 7000
# Spin up time steps
spinup_len = 2000
# Hence, in total:
ntotal = ntotal_clipped + spinup_len

# ICs
#X0=0.5
#Y0=1.5
#Z0=11.1

X0 = 1
Y0 = 1
Z0 = 1

# r = 166.08
r = 166.096
# pc = 99
pc = 98
nmembers = 25

# GENERATE OBS
Z = generate_timeseries(X0,Y0,Z0,ntotal,r)
Z_clipped = Z[spinup_len:]

pc_val_obs = N.percentile(Z_clipped,pc)

f = identify_features(Z_clipped,threshold=pc_val_obs,footprint=3,dx=1)





obs_exceed = N.where(Z_clipped > pc_val_obs,True,False)[N.newaxis,:]
# arr = N.vstack((N.arange(len(obs_exceed)),obs_exceed))
# obs_exceed = obs_exceed[N.newaxis,:]
# pdb.set_trace()
fig,ax = plt.subplots(1)
ax.pcolormesh(obs_exceed,cmap=M.cm.cubehelix_r)
fname = "test_exceed.png"
fpath = fname
fig.tight_layout()
fig.savefig(fpath)

# GENERATE GOOD ENSEMBLE forecast
# Needs to be done every x timesteps to create nultiple forecasts
# We leave the spinup time in the forecast, but shouldn't eval it

fcsts = N.empty([2,nmembers,ntotal+1])

mp_good = 0.00005
mp_bad = 0.005
icmulti = 1
for nm in range(nmembers):
    fcsts[0,nm,:] = generate_timeseries(X0,Y0,Z0,ntotal,r,
                                maxpert=mp_good,icmulti=icmulti)
    fcsts[1,nm,:] = generate_timeseries(X0,Y0,Z0,ntotal,r,
                                maxpert=mp_bad,icmulti=icmulti*3)

pc_val = dict()

for e in range(2):
    pc_val[e] = N.percentile(fcsts[e,:,:],pc)
    fcst_idxs = N.where(fcsts[e,:,:] > pc_val[e])
    pass

fig,axes = plt.subplots(nrows=3)

for n in range(3):
    ax = axes.flat[n]
    if n==0:
        ts = Z
    elif n==1:
        ts = fcsts[0,0,:]
    elif n==2:
        ts = fcsts[1,0,:]
    ax.plot(ts)

    # Plot percentile line (for only this forecast...)
    # Need to do new climo percentile over all fcst times
    pc_val = N.percentile(ts,pc)
    ax.axhline(pc_val,color='red',zorder=101,lw=0.5)
    # ax.set_xlim([0,ntotal])
    ax.set_ylim([0,300])

fname = "raw_ts.png"
fpath = fname
fig.tight_layout()
fig.savefig(fpath)

fcst_exceed = N.zeros_like(fcsts)

pc_val_fcst0 = N.percentile(fcsts[0,:,:],pc)
pc_val_fcst1 = N.percentile(fcsts[1,:,:],pc)
fcst_exceed[0,:,:] = N.where(fcsts[0,:,:] > pc_val_fcst0,
                        True,False)[N.newaxis,:]
fcst_exceed[1,:,:] = N.where(fcsts[1,:,:] > pc_val_fcst1,
                        True,False)[N.newaxis,:]

# idx x bool


#fcst_exceed0 = fcsts[0,fcst_exceed0_idx]
#fcst_exceed1 = fcsts[0,fcst_exceed1_idx]

# arr = N.vstack((N.arange(len(obs_exceed)),obs_exceed))
# obs_exceed = obs_exceed[N.newaxis,:]
# pdb.set_trace()
fig,axes = plt.subplots(3)
axes.flat[0].pcolormesh(obs_exceed,cmap=M.cm.cubehelix_r)
axes.flat[1].pcolormesh(fcst_exceed[:,0,:],cmap=M.cm.cubehelix_r)
axes.flat[2].pcolormesh(fcst_exceed[:,1,:],cmap=M.cm.cubehelix_r)

fname = "test_exceed_firstmembers.png"
fpath = fname
fig.tight_layout()
fig.savefig(fpath)

##################
members_count = N.arange(5)
ws = 150

for ne, ens in enumerate(("good","bad")):
    fig,axes = plt.subplots(nrows=members_count.size+2,figsize=(16,12))

    ax = axes.flat[0]
    ax.pcolormesh(obs_exceed,cmap=M.cm.Reds)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    max_fcst_ts = N.zeros([2,nmembers,ntotal_clipped])


    # Discretise ALL forecasts
    for nmem in range(nmembers):
    #for nmem in members_count:
        #max_fcst_ts[ne:ne+1,nmem,:] = maximum_filter(fcst_exceed[ne:ne+1,
        #                        nmem,spinup_len:-1],
        #                        size=ws,mode='constant')
        max_fcst_ts[ne:ne+1,nmem,:] = discretise_ts(fcst_exceed[ne:ne+1,
                                    nmem,spinup_len:-1],size=ws)
    # Plot probs
    probs_raw = N.sum(max_fcst_ts,axis=1)/nmembers

    # smooth them
    # probs = uniform_filter(probs_raw,size=100)
    probs = maximum_filter(probs_raw[ne:ne+1,:],size=ws,mode='constant')
    ax = axes.flat[1]
    #ax2 = ax.twinx()
    ax.pcolormesh(probs,cmap=M.cm.Blues,zorder=50)
    #ax2.plot(N.arange(ntotal_clipped),probs[0,:],color='black',lw=1,zorder=100)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    #ax2.axes.get_xaxis().set_visible(False)
    #ax2.axes.get_yaxis().set_visible(False)

    # ADD another axis here with "matched probs", where each event is
    # matched, if possible, within a tolerance.

    # for every True in the obs, locate the idx in the fcst field
    # check for a True in each fcst member within discretise size * 5
    # to simulate

    axs = axes.flat[2:]
    nnn = 0

    for nmem, nax in zip(members_count,axs):
        ax = nax
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        # newarray = N.where(fcsts[nmem,nnn,:] > pc_val)
        #max_fcst_ts =  maximum_filter(fcst_exceed[ne:ne+1,nmem,spinup_len:],
        #                        size=100,mode='constant')

        # ax.pcolormesh(fcst_exceed[ne:ne+1,nmem,spinup_len:],
        ax.pcolormesh(max_fcst_ts[ne:ne+1,nmem,:],
                cmap=M.cm.cubehelix_r)
        print("Plotted axis",nmem)

    fname = f"test_exceed_{ens}.png"
    fpath = fname
    fig.tight_layout()
    fig.savefig(fpath)


# Now evaluate each time period's probs. with a temporal tolerance

# VERIFICATION
# Only verify post-spinup period.
