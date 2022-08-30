"""XES score.
"""
import os
import pdb

import numpy as np

from scipy.stats import gmean as geomean

class CrossEntropy:
    def __init__(self,f,o,fk=None,unc=None,log_base=2,):
        """Compute the cross-entropy and optionally its components.

        An error will raise if binary values are passed in. Use the bounding
        staticmethod if you need to bound.

        Args:
            (copy from BrierScore)
        """
        self.f = np.array(f)
        self.o = np.array(o)

        # Check for binary values in forecast
        self.log_base = log_base
        if log_base is not 2:
            raise NotImplementedError

        # self.use_mean = "geometric"
        self.use_mean = "arithmetic"

        if fk is None:
            self.fk = np.unique(self.f)
        else:
            self.fk = fk
            raise NotImplementedError

    def do_mean(self,n):
        if self.use_mean == "geometric":
            f = geomean
        elif self.use_mean == "arithmetic":
            f = np.mean

        if isinstance(n,(float,int)):
            return n
        else:
            return f(n)

    @staticmethod
    def compute_DKL(x,y,return_all=False):
        """ Kullback-Liebler Divergence

        Args:
            x   : 1-D (e.g., observations)
            y   : 1-D (e.g., forecasts)

        The array with "atoms" (timesteps) is preserved for use in e.g. looking
        at individual events (however unrepresentative they can be).
        """
        if not isinstance(x,np.ndarray):
            x = np.array([x,])
        if not isinstance(y,np.ndarray):
            y = np.array([y,])

        with np.errstate(divide='ignore',invalid='ignore'):
            # Enforce array so we can remove nans
            term1 = (1-x) * np.log2((1-x)/(1-y))
            term2 = x * np.log2(x/y)

        term1[x==1] = 0
        term2[x==0] = 0

        DKL_all = term1 + term2
        DKL = np.mean(DKL_all)
        # print(f"DKL = {DKL:.4f}")

        if return_all:
            return DKL, DKL_all
        return DKL

    @staticmethod
    def _compute_xes(f,o):
        XES = np.mean(-((1-o)*np.log2(1-f)) - (o*np.log2(f)))
        return XES

    def get_xes(self,from_components=False,return_all=False,use_dkl=False):
        """ Cross-entropy score.
        """
        if from_components:
            REL = self.compute_rel()
            DSC = self.compute_dsc()
            UNC = self.compute_unc()
            XES = REL-DSC+UNC
        elif use_dkl:
            if return_all:
                XES, XES_all = self.compute_DKL(self.o,self.f,return_all=True)
            else:
                XES = self.compute_DKL(self.o,self.f)
        else:
            with np.errstate(divide='ignore',invalid='ignore'):
                XES = self._compute_xes(self.f,self.o)

        if return_all:
            return XES, XES_all
        return XES

    @staticmethod
    def _compute_entropy(x):
        return (-(1-x) * np.log2(1-x)) - (x * np.log2(x))

    @classmethod
    def get_entropy(cls,x,return_all=False):
        """ Average surprise. A series of values are returned if return_all.
        """
        with np.errstate(divide='ignore',invalid='ignore'):
            H_all = cls._compute_entropy(x)
        H = np.mean(H_all)
        if return_all:
            return H, H_all
        return H

    def compute_rel(self):
        rel_all = np.zeros_like(self.fk)
        for nk,k in enumerate(self.fk):
            ok = self.o[self.f==k]
            if ok.size != 0:
                ok_hat = self.do_mean(ok)
                rel_all[nk] = ok.size * self.compute_DKL(ok_hat,k)
        REL = np.sum(rel_all)/self.o.size
        print(f"{rel_all=}, REL = {REL:.4f}")
        return REL

    def compute_dsc(self):
        dsc_all = np.zeros_like(self.fk)
        for nk,k in enumerate(self.fk):
            ok = self.o[self.f==k]
            if ok.size != 0:
                ok_hat = self.do_mean(ok)
                o_hat = self.do_mean(self.o)
                dsc_all[nk] = ok.size * self.compute_DKL(ok_hat,o_hat)
        DSC = np.sum(dsc_all)/self.o.size
        print(f"{dsc_all=}, DSC = {DSC:.4f}")
        return DSC

    def __compute_UNC(self):
        """ Compute uncertainty component of the forecast.
        """
        p_o = self.do_mean(self.o)
        with np.errstate(divide='ignore',invalid='ignore'):
            term1 = -p_o*np.log2(p_o)
            term2 = -(1-p_o)*np.log2(1-p_o)
        UNC = term1 + term2
        print(f"UNC = {UNC:.4f}")
        return UNC

    def compute_UNC(self):
        p_o = self.do_mean(self.o)
        UNC = self.compute_entropy(p_o)
        print(f"UNC = {UNC:.4f}")
        return UNC

    @staticmethod
    def bound(x,thresh="auto",Ne=None):
        if (thresh == "auto"):
            thresh = 1/(3*Ne)
        else:
            assert isinstance(thresh,(float,np.float))
        x[x<thresh] = thresh
        x[x>(1-thresh)] = 1-thresh
        return x

    def compute_xess(self,from_components=False):
        UNC = self.compute_UNC()
        XES = self.compute_XES(from_components=from_components)
        XESS = (XES-UNC)/(0-UNC)
        return XESS

