"""Lorenz-63 toy model.
"""
import itertools
import os
import pdb

import matplotlib.pyplot as plt

import numpy as np

from plot.heatmaps import HeatMap

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
            print(f"The {pc}th percentile is {PCs[pc]:.1f}")
            plt.axvline(x=PCs[pc],zorder=100,color="black")
        return fig,ax

    @staticmethod
    def quantised_exceedence_timeseries(timeseries,pc,timewindow,figsize=(10,4),
                                            niceview=20e3,maxmin="max"):
        """Turn a time series into a quantised yes/no of
        exceedence over a given percentile. The time window is multiples of
        whatever the dt was (i.e., every time unit is 1 from data passed in)

        There is no tolerance in time (nonlocal) - this can be
        changed with varying time windows, but errors should equal out.
        For individual events, maybe more focussed fuzzy logic.
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
        fig,ax = plt.subplots(1,figsize=figsize)
        # plot_data = quant_data[:int(niceview/timewindow)]
        plot_data = gtlt(quant_data,np.ones_like(quant_data)*pc_val).astype(int)
        ax.plot(plot_data)
        plt.axhline(pc_val,)
        pass
        return fig, ax, plot_data

if __name__ == '__main__':
    plt.close("all")
    x0 = 1.0
    y0 = 1.0
    z0 = 1.0
    dt = 0.001
    rho = 166.08
    nt_raw = 204E3
    clip = 4E3
    nt = nt_raw - clip

    L63 = Lorenz63(x0,y0,z0,rho,dt=dt)
    L63.integrate(nt_raw,clip=clip)
    fig,ax = L63.plot_data()
    fig.show()

    fig,ax = L63.plot_spectrum()
    fig.show()

    # This is the raw integration with the spin-up clipped off.
    # Next a max/min filter over periods to create binary, intermittent dataset

    pc = 0.25
    tw = 20 # time window
    # For exceeding negative - more intermittent
    fig, ax, control_1d = L63.quantised_exceedence_timeseries(L63.get_z(),pc,
                                                           tw,maxmin="min")
    total = len(control_1d)
    # Aim for N,2N ratio for waffleplot
    control_2d = control_1d.reshape(100,-1)
    # fig.show()

    del fig,ax

    hm = HeatMap(control_2d,'../figures/heatmap_test.png')
    hm.plot(no_axis=True)
    hm.fig.show()
    pass

    # This is now "truth".

    # Let's do 10 similar alternate realities.
    # Each has a slight perturbation to the initial conditions
    rng = np.random.default_rng(27)
    # Not including truth
    nens = 8
    min_pert, max_pert = -1E-14,1E-14
    # Draw 30 small samples to add to the initial conditions
    perts = iter(rng.uniform(min_pert,max_pert,nens*3))

    pass
    # Rewrite to be in parallel optionally to quickly create the ensemble
    # Will need for 100-1000 members!
    # What about memory? Maybe we move to a server?


    # TODO: save data to disc if it doesn't exist.

    fig,axes = plt.subplots(nrows=3,ncols=3,figsize=(12,12))
    ensemble_members = ["c000",]+[f"p{n:03d}" for n in np.arange(1,nens+1)]
    ensemble = {}
    for ens_no, ax in zip(ensemble_members, axes.flat):
        print("Generating and plotting ensemble member",ens_no)
        if ens_no == "c000":
            data2d = control_2d
            ax.set_title("TRUTH")
        else:
            ens_L63 = Lorenz63(x0+next(perts), y0+next(perts),
                                    z0+next(perts),rho=rho
                                    )
            ens_L63.integrate(nt_raw,clip=clip)
            _fig,_ax,data1d = Lorenz63.quantised_exceedence_timeseries(
                                            ens_L63.get_z(),pc,tw,maxmin="min")
            data2d = data1d.reshape(100,-1)
            ensemble[ens_no] = data2d
            ax.set_title(ens_no)
        # Plot the heatmap within the multipanel plot
        hm = HeatMap(data2d, fig=fig, ax=ax)
        _fig, _ax = hm.plot(no_axis=True, save=False)
    fig.show()

    # Annotate little arrow showing start of dataset squares (origin?)

    # Could do KDE of each ensemble heatmap with prob overlaid over the truth

    # Next is testing windowing - schematic showing how binary is detected
    # a bit like FSS to avoid verifying below effective resolution of model.
    # Curse of dimensionality.

    # Rewrite asyncronously so the time series are put into dict/on disc
    # then plotted here


    # Now do windowing "the event occurs within x epochs (time steps) of t0"
    # We change this window size like FSS to find a scale of skill
    # Or like a square lasso round the waffleplot - can we catch a "yes"?

    # Adding ICs perts and per-timestep drift will make a wider ensemble

    # Try changing the rho


    # Try changing the percentiles.


    # Stretch goal: the temporal correlation and KS entropy growth?...


    # Investigate the trade-off between ensemble members and window size?


    # Trade off between DSC and REL to optimise forecast? Constant w.r.t rho?


