"""XES score.
"""
import os
import pdb

import numpy as np

from scipy.stats import gmean as geomean


class CrossEntropy:
    """ For computing and visualising skill scores for probabilistic forecast.
    Currently for two categories (binary) only, but with observational error.
    """
    def __init__(self,f,o,fk=None):
        """ Initialise the suite.
        Args:
        f:  1-D prob forecasts [0,1]. The probabilities should be bounded to
                  avoid divergence to infinity
        o:  1-D obs forecast [0,1], may or may not include obs uncertainty
        fk: 1-D array of probability levels. If None, compute automatically.
        """
        self.f = np.array(f)
        self.o = np.array(o)
        self.logbase = 2
        # self.use_mean = "geometric"
        self.use_mean = "arithmetic"

        if fk == None:
            self.fk = np.unique(self.f)
        else:
            raise Exception
            # NOT IMPLEMENTED
            # self.fk = fk

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

    def compute_XESS(self,from_components=False):
        UNC = self.compute_UNC()
        XES = self.compute_XES(from_components=from_components)
        XESS = (XES-UNC)/(0-UNC)
        return XESS

    @staticmethod
    def compute_entropy(x,return_all=False):
        """ Average surprise. A series of surprise
        values are returned if return_all is True.
        """
        with np.errstate(divide='ignore',invalid='ignore'):
            H_all = (-(1-x) * np.log2(1-x)) - (x * np.log2(x))
        H = np.mean(H_all)
        if return_all:
            return H, H_all
        return H

    def compute_XES(self,from_components=False,return_all=False,use_dkl=False):
        """ Cross-entropy score.

        Args:

        """
        if from_components:
            REL = self.compute_REL()
            DSC = self.compute_DSC()
            UNC = self.compute_UNC()
            XES = REL-DSC+UNC
        elif use_dkl:
            if return_all:
                XES, XES_all = self.compute_DKL(self.o,self.f,return_all=True)
            else:
                XES = self.compute_DKL(self.o,self.f)
        else:
            with np.errstate(divide='ignore',invalid='ignore'):
                XES = np.mean(-((1-self.o)*np.log2(1-self.f))-
                        (self.o*np.log2(self.f)))

        if return_all:
            return XES, XES_all
        return XES

    def compute_REL(self):
        rel = np.zeros_like(self.fk)
        for nk,k in enumerate(self.fk):
            ok = self.o[self.f==k]
            if ok.size != 0:
                ok_hat = self.do_mean(ok)
                rel[nk] = ok.size * self.compute_DKL(ok_hat,k)
        REL = np.sum(rel)/self.o.size
        print(f"{rel=}, REL = {REL:.4f}")
        return REL

    def compute_DSC(self):
        dsc = np.zeros_like(self.fk)
        for nk,k in enumerate(self.fk):
            ok = self.o[self.f==k]
            if ok.size != 0:
                ok_hat = self.do_mean(ok)
                o_hat = self.do_mean(self.o)
                dsc[nk] = ok.size * self.compute_DKL(ok_hat,o_hat)
        DSC = np.sum(dsc)/self.o.size
        print(f"{dsc=}, DSC = {DSC:.4f}")
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
