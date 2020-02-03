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
from scipy.ndimage.filters import maximum_filter1d
import numpy as N
import matplotlib as M
import matplotlib.pyplot as plt
from numpy import random
import skimage
from scipy import signal

overwrite_ts = True

def lorenz(x,y,z,sigma=10,b=2.667,r=28):
    xdot = sigma * (y-x)
    ydot = r*x - y - x*z
    zdot = x*y - b*z
    return xdot, ydot, zdot

def generate_timeseries(X0,Y0,Z0,nt,r,clip,maxpert=None,
                            discretise_size=None,icmulti=1):
    """This is clipped for spin-up, so
    first 1000 or so points are not "off the attractor".
    """
    X = N.empty(nt+1)
    Y = N.empty(nt+1)
    Z = N.empty(nt+1)

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
            # Encourage drift for bad ens
            #if pert < 0:
            #    pert = pert * icmulti

        X[n+1] = X[n] + (dt*xdot) + pert
        Y[n+1] = Y[n] + (dt*ydot) + pert
        Z[n+1] = Z[n] + (dt*zdot) + pert

    # print("Generated time series.")
    Z_clipped = Z[clip:]
    # pdb.set_trace()
    return Z_clipped

def generate_EPS(quality,nmembers,X0,Y0,Z0,nt,r,
                        clip,icmulti=1,):
    nt_clipped = nt-clip
    fcsts = N.zeros([nmembers,nt_clipped+1])

    base_mp = 0.00002
    multi = 100
    # max perturbation
    if quality == "good":
        mp = base_mp
        icmulti = 1
    else:
        # mp =  0.008
        mp = base_mp * multi
        icmulti = 10

    for nm in range(nmembers):
        fcsts[nm,:] = generate_timeseries(X0,Y0,Z0,nt,r,clip,
                                    maxpert=mp,icmulti=icmulti)
    return fcsts

def get_probs(arr,clip=True):
    p = N.sum(arr,axis=0)/arr.shape[0]
    return p

def generate_truth(X0,Y0,Z0,ntotal,r,clip):
    Z = generate_timeseries(X0,Y0,Z0,ntotal,r,clip)
    return Z

def discretise_ts(Z,size):
    nt = Z.size
    nblocks = int(nt/ws)+1
    Z_discrete = N.zeros(nblocks)

    posts = N.arange(Z.size-(size+1))[::size]
    for n, pn in enumerate(posts):
        if n != posts.size-1:
            idxn = n
            idxx = n+1 #posts[n+1]
            idx1n = posts[n]
            idx1x = posts[n+1]
            Z_discrete[idxn:idxx] = N.max(Z[idx1n:idx1x])
    # print("Discretised time series.")
    # return Z_discrete[N.newaxis,:]#.astype(bool)
    return Z_discrete#.astype(bool)

def identify_features(data,threshold,footprint,dx):
    """ Need to remember to cut times during spin-up time.
    """
    if data.ndim == 1:
        data = data[:,N.newaxis]

    data_masked = N.where(data >= threshold, data, 0)
    data_masked_bool = data_masked > 0.0
    return



#def FI_1D(fcst_array_2D,obs_array_1D,threshold,temporal_window):
def binarize(array,threshold):
    binary_arr = N.where(array > threshold,True,False)
    return binary_arr

def convolve(which,binary_arr,temporal_window):
    if which == "fcst":
        nens = binary_arr.shape[0]
        kernel = N.ones([2*nens,temporal_window])
        # kernel = N.ones([nens,temporal_window])
        # kernel = N.ones([1,temporal_window])
        total_voxels = temporal_window * nens
        # total_voxels = temporal_window
    elif which == "obs":
        kernel = N.ones([temporal_window])
        total_voxels = temporal_window

    # mode = "full"
    mode = "same"
    arr_conv = N.abs(N.around(signal.fftconvolve(
                            binary_arr,kernel,mode=mode)))

    # import pdb; pdb.set_trace()
    arr_conv = (arr_conv/total_voxels)
    # WHY IS THIS?...
    if which == 'fcst':
        arr_conv = arr_conv[0,:]
    assert N.all(arr_conv >= 0.0)
    assert N.all(arr_conv <= 1.0)
    return arr_conv

def remove_nans(val):
    if isinstance(val,N.ndarray):
        val[N.isnan(val)] = 0
        val[val == -N.inf] = 0
        val[val == -N.inf] = 0
    elif isinstance(val,float):
        if N.isnan(val) or val==N.inf or val==-N.inf:
            val = 0
    return val

def compute_REL(nk,ok,fk):
    """
    nk is number of forecasts in this prob bin, 2D
    ok_arr has value of o-bar for each prob bin: 2D
    fk_arr has prob value for each bin: 2D
    likewise, iok and ifk are for the not-observed values.
    """
    ok = remove_nans(ok)
    val_allk = nk*( (ok * N.log2(ok/fk)) + ( (1-ok) * N.log2((1-ok)/(1-fk)) ) )
    val = N.nansum(val_allk)/N.nansum(nk)
    # pdb.set_trace()
    return remove_nans(val)

def compute_RES(nk,ok,o):
    ok = remove_nans(ok)

    val_allk = nk*( (ok * N.log2(ok/o)) + ( (1-ok) * N.log2((1-ok)/(1-o)) ) )
    val = N.nansum(val_allk)/N.nansum(nk)
    # pdb.set_trace()
    return remove_nans(val)


def compute_UNC(o):
    val = -o*N.log2(o) - (1-o)*N.log2(1-o)
    return remove_nans(val)


def get_DKL(f,o):
    DKL = (1-o)*N.log2((1-o)/(1-f)) + o*N.log2(o/f)
    # pdb.set_trace()

    return remove_nans(DKL)

def init_ax(axes,n):
    ax = axes.flat[n]
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    return ax

def compute_SS(unc,dkl=None,rel=None,res=None):
    if dkl is not None:
        val = (N.mean(dkl)-unc)/(0-unc)
    return val


# temporal granularity of the integration?
dt = 0.0008

# Desired number of integration steps
ntotal_clipped = 10000
# Spin up time steps
spinup_len = 2000
# Hence, in total:
ntotal = ntotal_clipped + spinup_len

X0 = 1
Y0 = 1
Z0 = 1

# r = 166.08
r_base = 166.08
# pc = 99
pc = 99
nmembers = 10
ndays = 3

# GENERATE OBS OR TRUTH

# GENERATE 50x FCSTS EACH FOR GOOD AND BAD
# 2D, member x time

# Try to load npys
f_good_npyf = "f_good.npy"
f_bad_npyf = "f_bad.npy"
o_npyf = "o.npy"

if not os.path.exists(o_npyf) or overwrite_ts:
    print("Creating time series.")

    # If they don't exist:
    fcsts_good = N.zeros([ndays,nmembers,ntotal_clipped+1])
    fcsts_bad = N.zeros([ndays,nmembers,ntotal_clipped+1])
    obs_ts = N.zeros([ndays,ntotal_clipped+1])

    for day in range(ndays):
        random.seed(day+1)
        # perturbation in ICs just for this day
        dp = 0.1
        day_X = X0 + random.uniform(-dp,dp)
        day_Y = Y0 + random.uniform(-dp,dp)
        day_Z = Z0 + random.uniform(-dp,dp)

        rp = 0.0001
        day_r = r_base + random.uniform(-rp,rp)

        fcsts_good[day,:,:] = generate_EPS('good',nmembers,day_X,day_Y,day_Z,
                                ntotal,
                                clip=spinup_len,icmulti=1,r=day_r)
        fcsts_bad[day,:,:] = generate_EPS(
                                'bad',nmembers,day_X,day_Y,day_Z,ntotal,
                                clip=spinup_len,icmulti=3,r=day_r)
        obs_ts[day,:] = generate_truth(day_X,day_Y,day_Z,ntotal,clip=spinup_len,
                                r=day_r)
    # save
    N.save(file=f_good_npyf,arr=fcsts_good)
    N.save(file=f_bad_npyf,arr=fcsts_bad)
    N.save(file=o_npyf,arr=obs_ts)
else:
    print("Loading time series.")
    fcsts_good = N.load(f_good_npyf)
    fcsts_bad = N.load(f_bad_npyf)
    obs_ts = N.load(o_npyf)


# THRESHOLD
# Climates are the same pretty much
obs_thresh_pcval = N.percentile(obs_ts,pc)
fcst_thresh_pcval = obs_thresh_pcval

# Do thresholding
obs_exceed = N.where(obs_ts > obs_thresh_pcval,True,False)# [N.newaxis,:]
fcst_good_exceed = N.where(fcsts_good > fcst_thresh_pcval,True,False)
fcst_bad_exceed = N.where(fcsts_bad > fcst_thresh_pcval,True,False)

# DISCRETISE FCST
# Discretise both to mimic real-life snapshots in time for fcst/obs
# Done with max filter to generate realistic time series of 1/0s
# This isn't related to filtering in verification algorithm

# window size
ws = 150

obs_block = N.zeros([ndays,int(ntotal_clipped/ws)+1])
fcst_good_block = N.zeros([ndays,nmembers,int(ntotal_clipped/ws)+1])
fcst_bad_block = N.zeros([ndays,nmembers,int(ntotal_clipped/ws)+1])

# Generate "blocks" for each timestep
for day in range(ndays):
    obs_block[day,:] = discretise_ts(obs_exceed[day,:],size=ws)
    for nmem in range(nmembers):
        fcst_good_block[day,nmem,:] = discretise_ts(
                        fcst_good_exceed[day,nmem,:],size=ws)
        fcst_bad_block[day,nmem,:] = discretise_ts(
                        fcst_bad_exceed[day,nmem,:],size=ws)

# pdb.set_trace()
### So the blocks arrays are the NWP timescales, but otherwise "real life"


# VERIFICATION ################################

"""
# Do object-style here.
# For each time, create 5-tw blocks that are the max of each run + truth
# Then compute DKL per time step for RES, but REL and UNC need to be computed
# over the whole thing?
"""
# tw must be odd
tw = 3 # This is 2 either side of the object
obs_maxfilt = maximum_filter1d(obs_block,size=tw,axis=-1)
fg_maxfilt = maximum_filter1d(fcst_good_block,size=tw,axis=-1)
fb_maxfilt = maximum_filter1d(fcst_bad_block,size=tw,axis=-1)

x = int((tw-1)/2)
#
# obs_maxfilt[:,x:-x] = 0.001
# fg_maxfilt[:,x:-x] = 0.001
# fb_maxfilt[:,x:-x] = 0.001

fg_mf_probs = N.mean(fg_maxfilt,axis=1) # [day x prob series]
# fg_mf_probs[]

fb_mf_probs = N.mean(fb_maxfilt,axis=1)


# Bound!
def bound(arr,minthresh):
    arr[arr < minthresh] = minthresh
    arr[arr > 1-minthresh] = 1-minthresh
    return arr

# This thresh is what replaces a zero.
standard_thresh = (1/(3*nmembers))
# This can be set to a much smaller value to punish false alarms less
# thresh = standard_thresh * 0.1
thresh = standard_thresh

fg_mf_probs = bound(fg_mf_probs,thresh)
fb_mf_probs = bound(fb_mf_probs,thresh)

# To work out REL, RES, and UNC for the ensemble as a whole,
# we need to collect prob data
probbins_unbound = N.linspace(0,1,nmembers+1)
probbins = bound(probbins_unbound,thresh)
nprobbins = probbins.size
assert nprobbins == nmembers + 1
nt_block = fg_mf_probs.shape[1]

# data in here!
# first dimension, idx corresponds to:
# 1 = events that happened
# 0 = events that didn't happen

# DKL is a function of observed prob (1,0), which bin that goes in,
# for each day,
# for each timestep,
# put the observed prob in the right prob bin

# hist_good = N.zeros_like(probbins)
# hist_bad = N.zeros_like(hist_good)

nk_good = N.zeros_like(probbins)
nk_bad = N.zeros_like(probbins)

ok_good = N.zeros_like(probbins)
ok_bad = N.zeros_like(probbins)

o = 0

fk = probbins

ntimes = obs_maxfilt.shape[1]
bigN = (ntimes * ndays)
DKL_obj_good = N.zeros_like(obs_maxfilt)
DKL_obj_bad = N.zeros_like(DKL_obj_good)

RELs = N.zeros_like(obs_maxfilt)
RESs = N.zeros_like(obs_maxfilt)

#ooo = N.average(obs_maxfilt)

if False:
    for nday in range(ndays):
        for nt in range(nt_block):
            ob = obs_maxfilt[nday,nt]
            if ob == 1:
                ob = 0.999
            else:
                ob = 0.001

            good_prob = fg_mf_probs[nday,nt]
            bad_prob = fb_mf_probs[nday,nt]

            pbin_g_idx = N.where(probbins==good_prob)[0][0]
            pbin_b_idx = N.where(probbins==bad_prob)[0][0]

            nk_good[pbin_g_idx] += 1
            nk_bad[pbin_b_idx] +=1

            ok_good[pbin_g_idx] += ob
            ok_bad[pbin_b_idx] += ob

            o += ob

            DKL_obj_good[nday,nt] = get_DKL(good_prob,ob)
            DKL_obj_bad[nday,nt] = get_DKL(bad_prob,ob)
            # pdb.set_trace()

            ob_arr = N.zeros_like(probbins)
            ob_arr[pbin_g_idx] = ob

            n_arr = N.zeros_like(probbins)
            n_arr[pbin_g_idx] = 1

            gp_arr = N.zeros_like(probbins)
            gp_arr[pbin_g_idx] = good_prob

            # RELs[nday,nt] = compute_REL(n_arr,ob_arr,gp_arr)
            # RESs[nday,nt] = compute_RES(n_arr,ob_arr,ob_arr)

if True:
    # freq of occurrence for obs in whole dataset
    # o = N.mean(obs_maxfilt)
    oflat = obs_maxfilt.flatten()
    fgflat = fg_mf_probs.flatten()
    fbflat = fb_mf_probs.flatten()

    o = N.sum(oflat==1)/bigN
    for kidx,k in enumerate(probbins):
        kwhere_g = N.where(fgflat==k)
        nk_good[kidx] = kwhere_g[0].shape[0]

        # zz = fg_mf_probs[fg_mf_probs == k].sum()
        # ok_good[kidx] = obs_maxfilt[kwhere_g].sum()/nk_good[kidx]
        #ok_good[kidx] = N.mean(oflat[kwhere_g])
        ocount = oflat[kwhere_g]
        ok_good[kidx] = N.sum(ocount)/bigN
        # pdb.set_trace()
    # pdb.set_trace()


#nk_good = obs_maxfilt.shape[0]
#nk_bad = nk_good

#oo = o/bigN
# ok_good = ok_good/bigN
# ok_bad = ok_bad/bigN


# print("Mean RES for good is ",N.mean(RESs))
# print("Mean REL for good is ",N.mean(RELs))

# import pdb; pdb.set_trace()




# This gives you the rank histogram for surprise observed events (info gain)
# over all times, all days, so just a function of prob bin
# this is the total number of times, or "big N" in our paper

# observed freq of obs per prob bin...


# These values here are for "fk": the counts of observed occurrence in
# each prob band, o_k in the paper

# Frequency of events per prob band
#observed_good = N.mean(goodmatrix,axis=(0,1))
#observed_bad = N.mean(badmatrix,axis=(0,1))

# observed_good = N.sum(goodmatrix[:,:,:],axis=(0,1))/bigN
# observed_bad = N.sum(badmatrix[:,:,:],axis=(0,1))/bigN

# these are the obar total baserates from the paper
# obg_tot = N.mean(observed_good)

# obg_tot = N.sum(observed_good)
# obb_tot = N.sum(observed_bad)
#ver_k_good = N.sum(goodmatrix,axis=(0,1))
#over_k_bad = N.sum(badmatrix,axis=(0,1))
#obg_tot = N.mean(over_k_good)
#obb_tot = N.mean(over_k_bad)
#o = N.mean(obs_maxfilt,axis=(0,1))
#obg_tot = o
#obb_tot = o

# These are the same (right?) because obs have same number of obs
# assert obg_tot == obb_tot

### These matrices are now for not-observed time periods, and represent the
# (1-o) terms (analogues of above) in the DKL equations

# this is for non-observed events that are assumed not to happen (info loss)
#not_observed_good = N.sum(goodmatrix[0,:,:,:],axis=(0,1))/bigN
#not_observed_bad = N.sum(badmatrix[0,:,:,:],axis=(0,1))/bigN

#nobg_tot = N.sum(not_observed_good)
#nobb_tot = N.sum(not_observed_bad)
# assert nobg_tot == nobb_tot

N.set_printoptions(precision=3,suppress=True)


# nk = nk_good #(observed_good*bigN)# + not_observed_good*bigN).astype(int)
# ok = observed_good
# fk = probbins
#iok = not_observed_good
#ifk = (1-probbins)
# o = obg_tot
#io = nobg_tot

unc = compute_UNC(o)


print("UNC is",unc)

rel_good = compute_REL(nk_good,ok_good,fk)
res_good = compute_RES(nk_good,ok_good,o)
print("Good REL is",rel_good)
print("Good RES is",res_good)

dkl_good = rel_good-res_good+unc
print("Good DKL is",dkl_good)

ss_good = (res_good-rel_good)/unc
print("Skill score for GOOD is",ss_good)

# nk = nk_bad # (observed_bad*bigN) # + not_observed_bad*bigN).astype(int)
# ok = observed_bad
# fk = probbins
# iok = not_observed_bad
# ifk = probbqoins
# o = obb_tot
# io = nobb_tot # These are same as before. function of obs only.

rel_bad = compute_REL(nk_bad,ok_bad,fk)
res_bad = compute_RES(nk_bad,ok_bad,o)
print("Bad REL is",rel_bad)
print("Bad RES is",res_bad)

dkl_bad = rel_bad-res_bad+unc
print("Bad DKL is",dkl_bad)

ss_bad = (res_bad-rel_bad)/unc
print("Skill score for BAD is",ss_bad)

# assert 1==0

print("TEST Skill score for good is",compute_SS(unc,dkl_good))
print("TEST Skill score for bad is",compute_SS(unc,dkl_bad))




# eFSS-style:

# time_windows = [1,5,9,15,21]
time_windows = [5,]
BS_good = N.zeros_like(obs_block)
BS_bad = N.zeros_like(obs_block)

IGN_good = N.zeros_like(obs_block)
IGN_bad = N.zeros_like(obs_block)

fcst_good_all = N.zeros_like(obs_block)
fcst_bad_all = N.zeros_like(obs_block)
obs_all = N.zeros_like(obs_block)



for day in range(ndays):
    for tw in time_windows:
        # Filter (convolve)
        fcst_good_all[day,:] = convolve("fcst",fcst_good_block[day,:,:],
                            temporal_window=tw)
        fcst_bad_all[day,:] = convolve("fcst",fcst_bad_block[day,:,:],
                            temporal_window=tw)
        obs_all[day,:] = convolve("obs",obs_block[day,:],temporal_window=tw)

        # Deal with divergence to infinity
        f_good = fcst_good_all[day,:]
        f_bad = fcst_bad_all[day,:]
        obs = obs_all[day,:]

        f_good[f_good == 1] = 0.999
        f_good[f_good == 0] = 0.001

        f_bad[f_bad == 1] = 0.999
        f_bad[f_bad == 0] = 0.001

        obs[obs == 1] = 0.995
        obs[obs == 0] = 0.005

        # obs[obs == (1/tw)] = (1/tw) - 0.005*(1/tw)
        # obs[obs == 0] = 0 + 0.005*(1/tw)

        # f_good = f_good*tw
        # f_bad = f_bad*tw
        # obs = obs*tw

        # Score with BS and IGN
        BS_good[day,:] = (f_good-obs)**2
        BS_bad[day,:] = (f_bad-obs)**2

        IGN_good[day,:] = get_DKL(f_good,obs)
        IGN_bad[day,:] = get_DKL(f_bad,obs)
        # import pdb; pdb.set_trace()
        pass

##### TEST PLOTS
"""
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
"""
##################
#Plot truth, 1-w fcst probs, 5-tw fcst probs, and first 5 members

####



# day = 0
zzdayzz = [0,1]
for nday, day in enumerate([0,1]):
    for ne, ens in enumerate(("good","bad")):
        fig,axes = plt.subplots(nrows=9,figsize=(14,12))

        # Plot truth
        ax = init_ax(axes,0)
        ax.pcolormesh(obs_block[nday:nday+1,:],cmap=M.cm.Reds)

        ax = init_ax(axes,1)
        if ens == "good":
            probs2 = fcst_good_all[nday:nday+1,:]
        elif ens == "bad":
            probs2 = fcst_bad_all[nday:nday+1,:]
        ax.pcolormesh(probs2,cmap=M.cm.Blues,zorder=50,vmin=0,vmax=(1/tw))

        ax = init_ax(axes,2)
        if ens == "good":
            score = BS_good[nday:nday+1,:]
        elif ens == "bad":
            score = BS_bad[nday:nday+1,:]
        ax.pcolormesh(score,cmap=M.cm.Purples,zorder=50)
        print(f"BS for {ens} on day {day} is {N.sum(score)}")

        ax = init_ax(axes,3)
        if ens == "good":
            score = IGN_good[nday:nday+1,:]
        elif ens == "bad":
            score = IGN_bad[nday:nday+1,:]
        print(f"IGN for {ens} on day {day} is {N.sum(score)}")
        ax.pcolormesh(score,cmap=M.cm.Purples,zorder=50)

        o = obs_all[nday,:]
        H = -o*N.log2(o) - (1-o)*N.log2(1-o)
        print("Entropy for today is",N.sum(H))

        for aa in [4,5,6,7,8]:
            nmem = aa-4
            ax = init_ax(axes,aa)
            if ens == "good":
                data = fcst_good_block
            elif ens == "bad":
                data = fcst_bad_block
            ax.pcolormesh(data[nday:nday+1,:],
                    cmap=M.cm.cubehelix_r)

        fname = f"test_exceed_{ens}_day{zzdayzz[nday]}.png"
        fpath = fname
        fig.tight_layout()
        fig.savefig(fpath)

summed_days = N.sum(BS_good,axis=1)
average_bs = N.mean(summed_days)
print("Average BS for good model is", average_bs)

summed_days = N.sum(BS_bad,axis=1)
average_bs = N.mean(summed_days)
print("Average BS for bad model is", average_bs)

summed_days = N.sum(IGN_good,axis=1)
average_ign = N.mean(summed_days)
print("Average IGN for good model is", average_ign)

summed_days = N.sum(IGN_bad,axis=1)
average_ign = N.mean(summed_days)
print("Average IGN for bad model is", average_ign)

print("Skill score for good is",compute_SS(unc,IGN_good))
print("Skill score for bad is",compute_SS(unc,IGN_bad))

def compute_BSS():
    pass

unc_briar = 0
print("BSS good is",compute_BSS(unc_briar,BS_good))
# DO BSS here!!!!!!!

IG = IGN_good - IGN_bad
