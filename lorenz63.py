""" Lorenz 63 model as Python class.

Requirements:
    * Create datasets for analysis
    * Need to change ICs, perturbations (decorrelated?), regime (r) parameter

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
"""
import pdb, os
import random

import numpy as N


class Lorenz63:
    def __init__(self,sigma=10,b=2.667,r=28,clip_first=0):
        """
        Args:
            * clip_first (int): remove first n steps to avoid spin-up
        """
        self.sigma = sigma
        self.b = b
        self.r = r

        self.clip_first = clip_first

        def do_lorenz63(self,x,y,z,r=28.0,sigma=10.0,b=2.667):
        xdot = sigma * (y-x)
        ydot = r*x - y - x*z
        zdot = x*y - b*z
        return xdot, ydot, zdot

    def generate_timeseries(self,X0=0.0,Y0=1.0,Z0=0.0,nt=5000,
                                b=2.667,sigma=10.0,r=28.0,
                                clip=None,maxpert=None,discretise_size=None,
                                icmulti=1,
                                # dt=0.0008):
                                dt=1):
        """...
        []
        """
        if r is None:
            r = self.r
        if clip is None:
            r = self.clip

        X = N.empty(nt+1)
        Y = N.empty(nt+1)
        Z = N.empty(nt+1)

        # IC perturbations
        if not maxpert: # For truth
            pert = 0
        else:
            pert = random.uniform(-maxpert,maxpert)

        ICpert = pert * icmulti

        # JRL: why do we need to do this? Should we not start
        # at X0, Y0, Z0 as passed into the kwarg?

        # X[0] = X0 + ICpert
        # Y[0] = Y0 + ICpert
        # Z[0] = Z0 + ICpert

        for n in range(nt):
            xdot, ydot, zdot = self.do_lorenz63(X[n],Y[n],Z[n],r=r,sigma=sigma,
                                                b=b,)
            # Add model error Here

            if not maxpert: # For truth
                pert = 0
            else:
                pert = random.uniform(-maxpert,maxpert)
                # Encourage drift for bad ens
                #if pert < 0:
                #    pert = pert * icmulti

            # do we really need dt? should just count integrations?
            # Just a scaling of the vector in 3D space?! Hmm...
            X[n+1] = X[n] + (dt*xdot) + pert
            Y[n+1] = Y[n] + (dt*ydot) + pert
            Z[n+1] = Z[n] + (dt*zdot) + pert

        return Z


    def clip(self,ts):
        # Removes self.clip_first timesteps to avoid spin-up
        ts_clipped = ts[clip]
        return ts_clipped

    def generate_truth(self,):
        pass

    def generate_perturbated(self,):
        """ Generate a number of time series randomly perturbed (uniform)
        """
        pass

    self.
