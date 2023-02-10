"""Lorenz-63 toy model.
"""
import itertools
import multiprocessing
import os
import pdb
import pickle
import random
import time

import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import scipy

from plot.heatmaps import HeatMap
from src.brierscore import BrierScore
from src.crossentropy import CrossEntropy


class Lorenz63:
    def __init__(self,x0,y0,z0,rho,
                 sigma=10.0,beta=2.667,dt=1e-3):
        """Chop off spin-up time?

        TODO:
            * Remove spin-up time (transients)
            * Add perturbations (with random seed allowed as arg)
        the paper values use the intermittent regime

        rho
        166.08 is intermittent
        166.1 is the (more intermittent) variation
        28.0 is the lorenz default
        """
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.dt =dt
        self.t = 0.0

        # The data array in x,y,z. Time appended in axis=3
        self.output = np.zeros((3,1))
        self.output[:,0] = [x0,y0,z0]

    def dxdt(self,x,y):
        return self.sigma * (y-x)

    def dydt(self,x,y,z):
        return x * (self.rho - z) - y

    def dzdt(self,x,y,z):
        return (x*y) - (self.beta * z)

    def double_approx(self,x,y,z):
        # double approx help from github user kendixon
        dx1 = self.dxdt(x,y)*self.dt + x
        dy1 = self.dydt(x,y,z)*self.dt + y
        dz1 = self.dzdt(x,y,z)*self.dt + z

        dx2 = self.dxdt(dx1,dy1)
        dy2 = self.dydt(dx1,dy1,dz1)
        dz2 = self.dzdt(dx1,dy1,dz1)

        x2 = self.dt*(0.5*dx1 + 0.5*dx2) + x
        y2 = self.dt*(0.5*dy1 + 0.5*dy2) + y
        z2 = self.dt*(0.5*dz1 + 0.5*dz2) + z

        return x2,y2,z2

    def integrate_once(self,x=None,y=None,z=None):
        """
        Args:
            t0    : Initial time
            y0    : Intiial state
        """
        if x is None:
            x = self.output[0,-1]
            y = self.output[1,-1]
            z = self.output[2,-1]
        # print(x,y,z)
        ret = self.double_approx(x,y,z)
        # print(ret)
        new_time = np.expand_dims(np.array(ret),axis=1)
        return new_time

    def integrate(self,n,clip=None):
        # Time 0 is already done
        n = int(n)
        integrations = np.zeros((3,n))
        for n in range(n):
            if n == 0:
                next_data = np.array(self.output[:,0])
            else:
                next_data = self.integrate_once(**{s:v for s,v in
                                            zip(["x","y","z"],
                                            integrations[:,n-1])})
            # print(self.output.shape,next_data.shape)
            # self.output = np.concatenate((self.output,next_data),axis=1)
            integrations[:,n] = next_data.squeeze()
            # print(self.output)
        #print(self.output)
        self.output = np.concatenate((self.output,integrations),axis=1)
        if clip is not None:
            clip = int(clip)
            self.output = self.output[:,clip:]
            # cut_nt = nt - clip
        return

    def get_z(self,):
        return self.output[2,:]

    def get_exceedence_ts(self,pc,vrbl='z'):
        data = self.get_z()
        exceed = self.get_pc_exceed(data,pc)
        return exceed

    @staticmethod
    def get_pc_exceed(ts,pc):
        return ts > np.percentile(ts,pc)

    @staticmethod
    def do_windowing(ts,wsize):
        """
        Do we want a max filter to quantise timeseries?

        Args:
        ts    : time series of raw data.
        wsize : epoch window size in time steps
        """
        nt = ts.size
        epochs = np.zeros(int(nt/wsize)+1).astype(bool)
        widx = np.arange(0,nt,wsize)
        windows = np.zeros_like(ts).astype(bool)
        epoch_idx = np.arange(int(wsize/2),nt,wsize)
        for nc,cidx in enumerate(widx):
            idx0 = int(nc*wsize)
            idx1 = int(idx0 + wsize)
            this_window = ts[idx0:idx1]
            epochs[nc] = True in this_window
            windows[idx0:idx1] = bool(epochs[nc])
        return windows

    @classmethod
    def mask_and_window(cls,arr,pc,wsize):
        if arr.ndim == 1:
            nt = arr.size
            arr = arr.reshape(1,nt)
        pc_arr = cls.get_pc_exceed(arr,pc)
        window_ts = np.zeros_like(pc_arr)
        for ne in range(pc_arr.shape[0]):
            window_ts[ne,:] = cls.do_windowing(pc_arr[ne,:],wsize)
        # window_ts = cls.do_windowing(pc_ts,wsize)
        return window_ts

    def plot_data(self,figsize=(10,4),maxn=None):
        fig, ax = plt.subplots(1,figsize=figsize)
        if maxn is None:
            maxn = len(self.get_z())
        z_data = self.get_z()[:maxn]
        tsteps = np.arange(len(z_data))
        ax.plot(tsteps,z_data)
        fig.tight_layout()
        return fig, ax

    def plot_spectrum(self,figsize=(10,4)):
        fig, ax = plt.subplots(1,figsize=figsize)
        n, bins, patches = ax.hist(self.get_z(), 101,color="red",
                                   alpha=0.4)
        fig.tight_layout()
        PCs = {}
        for pc in [0.1,1,10,50,90,99,99.9]:
            PCs[pc] = np.percentile(self.get_z(),pc)
            # print(f"The {pc}th percentile is {PCs[pc]:.1f}")
            plt.axvline(x=PCs[pc],zorder=100,color="black")
        return fig,ax

    @staticmethod
    def quantised_exceedence_timeseries(timeseries,pc,timewindow,figsize=(10,4),
                                        niceview=20e3,maxmin="max",do_fig=True):
        """Turn a time series into a quantised yes/no of
        exceedence over a given percentile. The time window is multiples of
        whatever the dt was (i.e., every time unit is 1 from data passed in)

        There is no tolerance in time (nonlocal) - this can be
        changed with varying time windows, but errors should equal out.
        For individual events, maybe more focussed fuzzy logic.

        Using min - the motivation in L63 could be the risk of extremes of
        downdraughts during HCR days - potential danger to aviation in PBL
        """
        niceview = int(niceview)
        pc_val = np.percentile(timeseries,pc)

        if maxmin == "max":
            np_f = np.max
            gtlt = np.greater_equal
        elif maxmin == "min":
            np_f = np.min
            gtlt = np.less_equal
        else:
            raise Exception

        # Exceedence needs data to become maximum per timewindow, like max wind
        # The indices for each window
        # idxs = [[i,i+timewindow] for i in np.arange(len(timeseries))[::timewindow]]
        quant_data = np.zeros(int(len(timeseries)/timewindow))
        idxs = np.arange(0,len(timeseries)-timewindow,timewindow)
        for n,idx0 in enumerate(idxs):
            idx1 = idx0 + timewindow
            quant_data[n] = np_f(timeseries[idx0:idx1])
        plot_data = gtlt(quant_data,np.ones_like(quant_data)*pc_val).astype(int)
        if do_fig:
            fig,ax = plt.subplots(1,figsize=figsize)
            # plot_data = quant_data[:int(niceview/timewindow)]
            ax.plot(plot_data)
            plt.axhline(pc_val,)
        else:
            fig, ax = None, None
        return fig, ax, plot_data





if __name__ == '__main__':

    ### FUNCS
    def run_eval(results_df,pert_ex,probs_bound,control_1d):
        cross = CrossEntropy(probs_bound,control_1d)
        brier = BrierScore(probs_bound,control_1d)
        pass
        print("Computing XES components:")
        XES_UNC = cross.compute_unc()
        XES_REL = cross.compute_rel()
        XES_DSC = cross.compute_dsc()
        print("Computing BS components:")
        BS_UNC = brier.compute_unc()
        BS_REL = brier.compute_rel()
        BS_DSC = brier.compute_dsc()

        results_df.loc[pert_ex]["XES_REL"] = XES_REL
        results_df.loc[pert_ex]["XES_DSC"] = XES_DSC
        print(f"{XES_REL=}, {XES_DSC=}")

        XES = cross.compute_xes(from_components=False)
        XES2 = cross.compute_xes(from_components=True)
        print(f"{XES=:.3f},  {XES2=:.3f}")

        XESS = cross.compute_xess(from_components=False)
        XESS2 = cross.compute_xess(from_components=True)
        print(f"{XESS=:.3f},  {XESS2=:.3f}")

        results_df.loc[pert_ex]["BS_REL"] = BS_REL
        results_df.loc[pert_ex]["BS_DSC"] = BS_DSC
        print(f"{BS_REL=}, {BS_DSC=}")

        BS = brier.compute_bs(from_components=False)
        BS2 = brier.compute_bs(from_components=True)
        print(f"{BS=:.3f},  {BS2=:.3f}")

        BSS = brier.compute_bss(from_components=False)
        BSS2 = brier.compute_bss(from_components=True)
        print(f"{BSS=:.3f},  {BSS2=:.3f}")

        results_df.loc[pert_ex]["XES"] = XES
        results_df.loc[pert_ex]["XES2"] = XES2
        results_df.loc[pert_ex]["XESS"] = XESS
        results_df.loc[pert_ex]["XESS2"] = XESS2
        print(f"{XES=}, {XESS=}, {XESS2=}")

        results_df.loc[pert_ex]["BS"] = BS
        results_df.loc[pert_ex]["BS2"] = BS2
        results_df.loc[pert_ex]["BSS"] = BSS
        results_df.loc[pert_ex]["BSS2"] = BSS2
        print(f"{BS=}, {BSS=}, {BSS2=}")

        results_df.loc[pert_ex]["BS_UNC"] = BS_UNC
        results_df.loc[pert_ex]["XES_UNC"] = XES_UNC
        return results_df

    def bound_probs(fcst_arr,thresh):
        probs = np.mean(fcst_arr,axis=0)
        probs_bound = np.copy(probs)
        probs_bound[probs_bound<thresh] = thresh
        probs_bound[probs_bound>1-thresh] = 1-thresh
        return probs_bound

    def do_tw(arr,kernel_tw,min_thresh,check_nans=True):
        # Check for nans?

        # This is a maximum applied to a convolution as a rolling max
        # "Event within X epochs within the time window"
        conv_ts = np.convolve(arr,np.ones([kernel_tw,]),mode="same"
                                            ) > kernel_tw*min_thresh
        pass
        # Trying to work out the weird TRUTH panel in plots for FSS-style tw
        if np.nan in conv_ts:
            assert True is False
        return conv_ts.astype(int)

    def asymmetrize(arr,lb=0.0005,ub=0.98):
        arr[arr>ub] = ub
        arr[arr<lb] = lb
        pass
        return arr

    def get_new_xyz(x,y,z,pert_size,rng):
        perts = rng.uniform(-pert_size,pert_size,3)
        return x+perts[0], y+perts[1], z+perts[2]

    def create_data_fpath(pert_ex,mode=""):
        if mode != "":
            mode = f"_{mode}"
        data_dir = '../data/data_2023'
        data_fname = f"test_E{pert_ex:.3f}{mode}.npy"
        return os.path.join(data_dir,data_fname)

    def plot_verif_results(df):
        fig,axes = plt.subplots(ncols=2, nrows=3, figsize=(10,8))

        def do_ss(ax,sc_type):
            if sc_type == "XES":
                sc_name = "XES"
                ss_name = 'XESS'
                sc_color = "maroon"
                ss_color = "indianred"
                # ls = '_'
                # ylim_sc = [0.0,0.16]
                ylim_sc = None
            elif sc_type == "BS":
                sc_name = "BS"
                ss_name = "BSS"
                sc_color = "darkblue"
                ss_color = "lightblue"
                # ls = '--'
                # ylim_sc = [0.0,0.016]
                ylim_sc = None
            else:
                raise Exception

            ylim_ss = [-1,1]
            lw = 0.75

            ax.scatter(df.index,df[sc_name],marker='x',label=sc_type,color=sc_color)
            ax.plot(df.index,df[sc_name],lw=lw,color=sc_color)
            ax.set_ylim(ylim_sc)
            ax.set_ylabel(sc_name)
            ax.legend(loc=2)

            ax2 = ax.twinx()
            ax2.axhline(linestyle='--',y=0,lw=1,color=ss_color)
            ax2.scatter(df.index,df[ss_name],marker="o",label=ss_name,color=ss_color)
            ax2.plot(df.index,df[ss_name],lw=lw,color=ss_color)
            ax2.set_ylim(ylim_ss)
            ax2.legend(loc=1)
            ax.set_title(sc_type)

            return ax

        def do_comps(ax,sc_type,comp_type):
            assert comp_type in ("REL","DSC")
            if sc_type == "XES":
                sc_name = "XES"
                dsc_color = "orangered"
                rel_color = "peru"
                # ylim_dsc = [0.0,0.12]
                # ylim_rel = [0.0,0.04]
                ylim_dsc = None
                ylim_rel = None
            elif sc_type == "BS":
                sc_name = "BS"
                dsc_color = "cadetblue"
                rel_color = "steelblue"
                # ylim_dsc = [0.0,0.013]
                # ylim_rel = [0.0,0.0015]
                ylim_dsc = None
                ylim_rel = None
            else:
                raise Exception

            ylim_ss = [-1,1]
            lw = 0.75
            comp_str = f"{sc_name}_{comp_type}"
            unc_str = f"{sc_name}_UNC"

            if comp_type == "DSC":
                ylim_sc = ylim_dsc
                sc_color = dsc_color
            else:
                sc_color = rel_color
                ylim_sc = ylim_rel

            ax.scatter(df.index,df[comp_str],marker='x',label=comp_str,color=sc_color)
            ax.plot(df.index,df[comp_str],lw=lw,color=sc_color)
            ax.set_ylim(ylim_sc)
            ax.set_ylabel(sc_name)
            ax.legend(loc=2)

            ax2 = ax.twinx()
            ax2.axhline(linestyle='dotted',y=0,lw=1.5,color=sc_color)
            ax2.scatter(df.index,df[comp_str]/df[unc_str],marker="o",color=sc_color,
                        alpha=0.5)
            ax2.plot(df.index,df[comp_str]/df[unc_str],linestyle='dashed',lw=lw,
                                                                color=sc_color)
            ax2.set_ylim(ylim_ss)
            ax2.legend(loc=1)
            ax.set_title(f"{sc_name} {comp_type} and its SS")

            return ax

        ax = axes[0][0]
        ax = do_ss(ax,"XES")

        ax = axes[0][1]
        ax = do_ss(ax,'BS')

        ax = axes[1][0]
        ax = do_comps(ax,"XES","DSC")

        ax = axes[1][1]
        ax = do_comps(ax,"BS","DSC")

        ax = axes[2][0]
        ax = do_comps(ax,"XES","REL")

        ax = axes[2][1]
        ax = do_comps(ax,"BS","REL")

        fig.tight_layout()
        fig.show()

    def generate_2d(arr,sq_len):
        data_2d = arr[:int(sq_len*sq_len)].reshape(sq_len,sq_len)
        pass
        return data_2d


    ### 4th Jan: to-do

    np.set_printoptions(precision=4)
    np.set_printoptions(suppress=True)
    x0 = 1.001
    y0 = 1.001
    z0 = 1.001
    dt = 0.001
    # rho = 166.08
    # rho = 166.088
    # rho = 166.09
    # rho = 166.1
    # rho = 166.085

    # Manneville and Pomeau have range of r values
    rho = 166.2

    ### Number of experiments ("days") to run this on
    ### Then concat-ensemble is the n experiments in one long time series
    # (Ignoring edge effects that are rare)
    # (Will have to consider loss
    nexps = 10 # 25
    nt_raw = 95E3 # 110E3
    clip = 5E3
    # clip = 500
    nt = nt_raw - clip
    overwrite = 1
    overwrite_tw = 1
    npert = 50
    nsubmem = 20
    # FSS-style window, +/- either side:
    # Eventually this will be a variable?
    # conv_tws = np.arange(5,30,4)
    conv_tw = 11

    pc = 1
    tw = 20 # time window for initial L63 timeseries - not FSS!
    total_epochs = int(nt/tw)
    rng = np.random.default_rng(27)
    # pert_exs = np.arange(-16.5,-13.05,0.2,dtype=float)
    pert_exs = np.arange(-18.0,-7.9,0.5)

    min_thresh_f = 0.001
    min_thresh_o = 0.0005
    max_thresh_o = 0.97
    sq_len = 50
    sc_names = ["XES","XES2","XESS","XESS2","BS","BS2","BSS","BSS2","XES_DSC",
                        "XES_REL","BS_DSC","BS_REL","XES_UNC","BS_UNC"]

    ### For changing truth - needs to be bigger
    # Each day should differ and not repeat!
    # Sample +/- 0.1 from the original x0 (later _x0)
    exp_pert = 0.001

    ##################################################################
    ##################################################################
    ##################################################################
    ## CREATE TRUTH ##
    # Writing this function here to be lazy so it inherits a load of vrbls
    def create_truth(x0,y0,z0,do_plot=False,do_obs_error=True,perturb=None,
                                        rng=None):
        if perturb is not None:
            x0, y0, z0 = get_new_xyz(x0,y0,z0,perturb,rng)

        L63 = Lorenz63(x0,y0,z0,rho,dt=dt)
        L63.integrate(nt_raw,clip=clip)
        if do_plot:
            fig,ax = L63.plot_data()
            fig,ax = L63.plot_spectrum()

        # This is the raw integration with the spin-up clipped off.
        # Next a max/min filter over periods to create binary, intermittent dataset

        # For exceeding negative - more intermittent
        fig, ax, control_1d = L63.quantised_exceedence_timeseries(L63.get_z(),pc,
                                                               tw,maxmin="min")

        ### Add observational error!
        if do_obs_error:
            control_1d = asymmetrize(np.copy(control_1d.astype(float)),
                                         lb=min_thresh_o, ub=max_thresh_o)
        return control_1d

    ##################################################################
    ##################################################################
    ##################################################################

    ## TODO JRL TO DO
    # The problem is truth must be created once per experiment
    # and put into ensemble as 0th member.

    # Start by creating truths (exps) and plotting first 4 in panel plot
    # First we need a list of lists of x0, y0, z0 for each exp

    # Save x/y/z as base truth then perturb round for each experiment, then
    # perturb round THAT for an ensemble with much smaller perts.

    xs = []
    ys = []
    zs = []
    for exp in range(nexps):
        _x, _y, _z = get_new_xyz(x0,y0,z0,exp_pert,rng)
        xs.append(_x)
        ys.append(_y)
        zs.append(_z)
    EXP_XYZ = {n:(x,y,z) for n,(x,y,z) in enumerate(zip(xs,ys,zs))}

    # Loop over each new x0/y0/z0 and create the truth
    TRUTH = dict()
    for nexp,exp in enumerate(range(nexps)):
        TRUTH[exp] = create_truth(*EXP_XYZ[exp],perturb=None)

    # Concatenated truths
    concat_len = total_epochs * nexps
    truth_arr_cc = np.zeros([concat_len])
    total_epochs = int(nt/tw)
    for nexp, exp in enumerate(np.arange(nexps,dtype=int)):
        truth_arr_cc[(nexp*total_epochs):(nexp+1)*total_epochs] = TRUTH[exp]

    ##################################################################
    ##################################################################
    ##################################################################

    control_1d = create_truth(x0,y0,z0,do_plot=True)
    control_2d = generate_2d(control_1d,sq_len)

    hm = HeatMap(control_2d,'../figures/heatmap_test.png')
    hm.plot(no_axis=True)
    hm.ax.set_title("Example truth")
    hm.fig.show()
    pass

    _x0 = np.copy(x0)
    _y0 = np.copy(y0)
    _z0 = np.copy(z0)

    # Think about Phoenix, AZ
    # We already don't reward persistence bu dividing by UNC.
    # But there's a lot of uncertainty about whether a tornado happened.
    # But there isn't about blue sky days
    # Verification is a lot harder for 1 than 0, so we shouldn't let
    # all the 0s build up and be very sensitive to the bounding probability.
    # So there should be a bigger handicap for whether a tornado happened
    # "Class error" not obs error.

    # Optimising an ensemble using information theory - AMS submission
    # (this one is mainly XES v BS and why include obs error and interp's.)
    # Wx Challenge - just me
    # Bigger picture about decoding the atmosphere -


    # Let's do 10 similar alternate realities.
    # Each has a slight perturbation to the initial conditions
    # Not including truth
    nmembers = npert + 1
    # Is uncertainty constant - just the control?

    results = np.zeros([len(pert_exs),len(sc_names)+1])
    results[:,0] = pert_exs
    results_df = pd.DataFrame(results,columns=["pert_size",]+sc_names)
    results_df.set_index("pert_size",inplace=True)
    # results_df.index = results_df.index.map(str.format,"%.1f")

    # Could we do this in parallel

    for _nper, pert_ex in enumerate(pert_exs):
        print("Doing experiment for pert size",pert_ex)
        max_pert = 10**pert_ex

        # TODO: Check what perts looks like, so we can make multiprocessed.
        pass
        # Rewrite to be in parallel optionally to quickly create the ensemble
        # Will need for 100-1000 members!
        # What about memory? Maybe we move to a server?


        # total_epochs = int(nt/tw)
        pass
        # TODO: save data to disc if it doesn't exist.
        # Plot the first five members plus truth
        fig,axes = plt.subplots(nrows=2,ncols=3,figsize=(10,8))
        # fig,axes = plt.subplots(nrows=3,ncols=4,figsize=(14,12))
        pert_members = [f"p{int(n):03d}" for n in np.arange(1,npert+1)]
        ensemble_members = ["c000",]+ pert_members

        data_fpath = create_data_fpath(pert_ex)
        pass
        probs_arr = np.zeros([nexps,total_epochs])

        if (not overwrite) and os.path.exists(data_fpath):
            print("Loading ensemble data from file.")
            # Concatenated ensemble
            cc_ensemble = np.load(data_fpath)
            probs = np.load(data_fpath.replace("test","probs"))
        else:
            ensemble = np.zeros([nexps,nmembers,total_epochs])
            for nexp, exp in enumerate(np.arange(nexps)):
            # EXP_XYZ = {n:(x,y,z) for n,(x,y,z) in enumerate((xs,ys,zs))}
            # for nexp, exp in enumerate()
                exp_x, exp_y, exp_z = EXP_XYZ[nexp]
                print("Doing experiment/day",nexp+1)

                # Draws samples to add to the initial conditions (new draw for each exp)
                # A new TRUTH must be created for each
                # A new experiment needs a new draw of x0, too
                # Like centering a cone of uncertainty
                ### THESE ARE FOR THIS EXP
                # x0, y0, z0 = get_new_xyz(_x0,_y0,_z0,exp_pert,rng)
                # exp_truth = create_truth(x0,y0,z0,do_plot=True)
                perts = iter(rng.uniform(-max_pert,max_pert,npert*3))

                for (nens, member_name), ax in itertools.zip_longest(
                                                    enumerate(ensemble_members),
                                                    axes.flat):
                    print("Generating and plotting ensemble member",member_name)
                    if nens == 0:
                        # We have truth - drop in here
                        data2d = generate_2d(TRUTH[nexp],sq_len)
                        ensemble[nexp,0,:] = TRUTH[nexp]
                        if (nens < 6) and (nexp == 0):
                            ax.set_title("TRUTH")
                    else:
                        mem_x = exp_x + next(perts)
                        mem_y = exp_y + next(perts)
                        mem_z = exp_z + next(perts)
                        ens_L63 = Lorenz63(mem_x, mem_y, mem_z, rho=rho)
                        ens_L63.integrate(nt_raw,clip=clip)
                        if (nens < 6) and (nexp == 0):
                            ax.set_title(member_name)
                            do_fig = True
                        else:
                            do_fig = False
                        _fig,_ax,data1d = Lorenz63.quantised_exceedence_timeseries(
                                                        ens_L63.get_z(),pc,tw,
                                                        maxmin="min",do_fig=do_fig)

                        # data2d = control_1d[:int(sq_len*sq_len)].reshape(sq_len,sq_len)
                        data2d = generate_2d(data1d,sq_len)
                        pass
                        ensemble[nexp,nens,:] = data1d
                        pass
                    # Plot the heatmap within the multipanel plot
                    probs_arr[nexp,:] = np.mean(ensemble[nexp,1:,:],axis=0)
                    if (nens < 6) and (nexp == 0):
                        hm = HeatMap(data2d, fig=fig, ax=ax)
                        _fig, _ax = hm.plot(no_axis=True, save=False)
                pass
                        # plt.close(_fig)
            fig.suptitle(f"Plots for pert size {pert_ex:.3f}")
            fig.show()

            # Need to check probs
            pass

            concat_len = total_epochs * nexps
            cc_ensemble = np.zeros([nmembers,concat_len])
            probs = np.zeros([concat_len])
            for nexp, exp in enumerate(np.arange(nexps)):
                for nens, ens in enumerate(np.arange(npert+1)):
                    # nens == 0 is the control (obs)
                    cc_ensemble[nens,(nexp*total_epochs):(
                            (nexp+1)*total_epochs)] = ensemble[nexp,nens,:]
                    # Probs
                probs[(nexp*total_epochs):(nexp+1)*total_epochs] = probs_arr[nexp,:]
            pass
            print("The unique probabilities are:", np.unique(probs))

            np.save(data_fpath,cc_ensemble)
            np.save(data_fpath.replace("test","probs"),probs)

        # Introduce obs error?
        # It's important because 0 and 1 have very different errors
        # We can be sure there was no tornado on a blue-sky day
        # But we can't be sure there was no tornado during a DMC day


        # if (-14.75 < pert_ex < -14.55):
        #     fig.show()
        # Probabilities

        # We should look at sensitivity to this


        # probs = np.mean(ensemble[1:],axis=0)
        probs_bound = CrossEntropy.bound(np.copy(probs),thresh=min_thresh_f)

        # probs_bound = np.copy(probs)
        # probs_bound[probs_bound<thresh] = thresh
        # probs_bound[probs_bound>1-thresh] = 1-thresh
        pass

        run_eval(results_df,pert_ex,probs_bound,cc_ensemble[0,:])

        # Check control_1d, probs_bound, probs... why only four levels?!
        pass

        # Test ignorance with tw=1
        # dkl = scipy.special.kl_div(probs_bound,control_1d)
        # dkl = scipy.special.kl_div(control_1d,probs_bound)
        # verif = scipy.special.rel_entr(control_1d,probs_bound)

    pass

    # This figure (results_df) shows how the ensemble sweet-spot for perts
    # is around 15.


    # Label with "no tornadoes detected", "some",
    # "information-transfer saturation"
    plot_verif_results(results_df)

    ############################################################
    ################## Now for time tolerance ##################
    ############################################################

    # FSS-like time windows (different from creating Lorenz time series)
    # Is there an event within 1,3,5,7... epochs for each time?
    # Need to convolve each time series
    results_tw = np.zeros([len(pert_exs),1+len(sc_names)])
    results_tw[:,0] = pert_exs
    # Eventually we want BS, BSS, XES, XESS, DSC, REL
    # Multi index?
    results_tw_df = pd.DataFrame(results_tw,columns=["pert_size",]+sc_names)
    results_tw_df.set_index("pert_size",inplace=True)

    # Need to rename original tw!
    del tw

    for npt, pert_ex in enumerate(pert_exs):
        fig,axes = plt.subplots(nrows=2,ncols=3,figsize=(10,8))
        data_fpath_tw = create_data_fpath(pert_ex,mode=f"{conv_tw:02d}tw")

        # ensemble data for this pert_ex
        data_fpath = create_data_fpath(pert_ex)
        cc_ensemble = np.load(data_fpath)

        if (not overwrite_tw) and os.path.exists(data_fpath_tw):
            print("Loading ensemble data from file.")
            smoothed = np.load(data_fpath_tw)
        else:
            smoothed = np.zeros([nmembers,nexps*total_epochs])
            for (nens, member_name), ax in itertools.zip_longest(
                    enumerate(ensemble_members),
                    axes.flat):
                print("Convolving ensemble member",member_name)
                if nens == 0:
                    # obs data needs to be independent of npt
                    # conv_ts = np.convolve(control_1d,np.ones([tw,]),mode="same") > 0
                    conv_ts = do_tw(cc_ensemble[0,:],conv_tw,min_thresh_o)
                    # data2d = conv_ts[:int(sq_len*sq_len)].reshape(sq_len,sq_len)
                    data2d = generate_2d(conv_ts,sq_len)
                    smoothed[0,:] = conv_ts
                    if nens < 6:
                        ax.set_title("TRUTH")
                        hm = HeatMap(data2d, fig=fig, ax=ax)
                        _fig, _ax = hm.plot(no_axis=True, save=False)
                    pass
                else:
                    # conv_ts = np.convolve(ensemble[nens,:],np.ones([tw,]),mode="same") > 0
                    conv_ts = do_tw(cc_ensemble[nens,:],conv_tw,min_thresh_f)
                    # data2d = conv_ts[:int(sq_len*sq_len)].reshape(sq_len,sq_len)
                    data2d = generate_2d(conv_ts,sq_len)
                    smoothed[nens,:] = conv_ts
                    if nens < 6:
                        ax.set_title(member_name)
                        hm = HeatMap(data2d, fig=fig, ax=ax)
                        _fig, _ax = hm.plot(no_axis=True, save=False)
            np.save(data_fpath_tw,smoothed)
            fig.show()

        probs_bound = bound_probs(smoothed[1:],min_thresh_f)
        run_eval(results_tw_df,pert_ex,probs_bound,smoothed[0])

    plot_verif_results(results_tw_df)

    # The reader can think of the heatmap plots as either going from the
    # bottom left and working up row-by-row as a time series, or simply
    # as matching the image like a png-png. Verification doesn't care
    # if the skill drops off exponentially, it is additive and irrespective
    # of time.

    # HERE - PUT DSC/REL

    # Not sure whether to use tw=1 or tw=tw for next bit
    # Let's use the smoothed to test

    # Let's run some things again with a subset of the ensemble members
    # Using x random plus control
    sub_idx_all = np.arange(npert)
    # TESTING
    sub_idx = random.sample(list(sub_idx_all[1:]),nsubmem)

    assert len(sub_idx) == nsubmem

    results_sub = np.zeros([len(pert_exs),len(sc_names)+1])
    results_sub[:,0] = pert_exs
    # Eventually we want BS, BSS, XES, XESS, DSC, REL (print UNC in subtitle)
    # Multi index?
    results_sub_df = pd.DataFrame(results_sub,columns=["pert_size"]+sc_names)
    results_sub_df.set_index("pert_size",inplace=True)

    for pert_ex in pert_exs:
        data_fpath = create_data_fpath(pert_ex)
        ens = np.load(data_fpath)
        sub_ens = ens[sub_idx,:]
        probs_bound_sub = bound_probs(sub_ens[1:],min_thresh_f)
        run_eval(results_sub_df,pert_ex,probs_bound_sub,sub_ens[0])

    plot_verif_results(results_sub_df)

    print("Percent of times with an event: (COMPUTE!)")

    #### Plot as a function of time! Tune the running time to make sure the
    #### scores aren't too low (signal overwhelmed by the randomness after
    #### information horizon)

    # bar charts of "dsc_all" to show how much information, in bits, has
    # been gained cumulatively per percent (k). This can be normalised
    # (DSC/UNC) - show this too.

    # This is all following a logistic-like curve (?)
    # What is the importance of the inflexion point?
    # Let's do REL and DSC.

    # Test it works!

    ### TO DO: maybe we need drift after every x timesteps
    ### change max_pert sampling to be more focussed on the changing bits
    ## maybe FSS-style time window needs to be larger
    ## this shows the time/length scale at which there is "skill"?
    ## This will only work if there are slight offsets with the Lorenz model
    ## Maybe the smoothing is too much on the quantisation - try 10 not 20?

    ### TO DO: add obs error so that correct 0s are not rewarded as highly

    # TO DO: plot k vs dsc/rel to show how model might be performing

    ### TO DO: individual analysis of each "event":
    ### Loop through start of each event, and verify each event

    # Annotate little arrow showing start of dataset squares (origin?)

    # Could do KDE of each ensemble heatmap with prob overlaid over the truth

    # Try changing the rho

    # Try changing the percentiles.

    # Change the number of ensemble members (subsample from 50)

    # Stretch goal: the temporal correlation and KS entropy growth?...


    # Investigate the trade-off between ensemble members and window size?


    # Trade off between DSC and REL to optimise forecast? Constant w.r.t rho?



    #### How to do the time windowing to avoid curse of dimensionality
    # For each time, the x-width window centred on it is True if an event
    # occurs within that window.


    # Could do mutual information of truth/ensemble over time to show drift
    # (exponential?)
    # Also show surprise-per-time-window? How to zoom in on one event
    # What about power spectrum or proof of intermittency like Manneville?
    # Error over time

    # Show why BSS doesn't have same XES estimate
    # How much is obs error (do independently)
    # How much is rarity of event
    # Try rarer event
    # IG doesn't care about being off by one class/epoch, it is local and
    # therefore not affected by intermittency. It is affected by rare
    # events, however, or low-probability in the ensemble. Looking at 1%
    # baserate in some ensembles (zoom in on period of forecast that
    # has low probs and look at difference in reward from IGN and BS)

    # (Run 100 members? Else we can't show small probability differences)

    # If too long of the time series past predictability limit is evaluated,
    # it only "waters" down the skill score value

    # We can estimate predictability limit at this rho and max_pert!

    # Look at mutual info of ensembles at high pert_size - are they clustered
    # away from truth, or do they drift "symmetrically"?

    # Could do difference between XESS/BSS over time on one graph
    # Averaging over each experiment in chunks of 10% of time series length?
    # Also mark where the zero line (no skill) is passed for each score