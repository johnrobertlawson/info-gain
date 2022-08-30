import pdb

import numpy as np
import pandas as pd

# How much of this can we get rid of?
np.set_printoptions(precision=4,suppress=True)
pd.set_option("display.precision",4)
pd.set_option("display.float_format",'{:.4f}'.format)

class BrierScore:
    def __init__(self,f,o,fk=None,unc=None):
        """Compute the Brier Score and optionally components.

        Args:
            f:      forecast array
            o:      observation array
            fk:     binning levels. If None, set automatically.
            unc:    if a float, manually set the uncertainty components to
                this value. If None, set using the variation of the data.
                This also allows a skill score to be formed (a la Lawson
                et al 2021) from one data point (e.g., 10% blanket probs).
        """
        self.o = np.array(o)
        self.f = np.array(f)

        if fk is None:
            self.fk = np.unique(self.f)
        else:
            self.fk = fk
            # Need to test this works if we pass manual bins
            # (e.g., the bins are too small for the dataset by default)
            # maybe raise warning if there's too few (<100?) to compute
            # stats for automatic bins.
            raise NotImplementedError

        self.unc = unc

    @staticmethod
    def do_mean(x):
        """Custom mean function due to arithmetic v geometric mean.
        """
        if isinstance(x,(float,int)):
            return x
        # Duck-typed so that collections are averaged; it breaks otherwise
        else:
            return np.mean(x)

    @staticmethod
    def _compute_bs(f,o):
        """Compute the Brier Score. This is for devs; users should use get_bs().

        Args:
            f,o (int, float, numpy.array):  forecast and obs fraction, resp.
        """
        return np.sum((f-o)**2)

    def get_bs(self,from_components=False):
        if from_components:
            REL = self.compute_REL()
            DSC = self.compute_DSC()
            UNC = self.compute_UNC()
            return REL - DSC + UNC
        else:
            return self._compute_bs(self.f,self.o)

    def compute_unc(self):
        # For testing
        obs_unc = True
        if obs_unc:
            # Including observational error
            UNC = np.sum((self.o-self.do_mean(self.o))**2)/self.o.size
        else:
            # Not...
            UNC = self.do_mean(self.o) * (1-self.do_mean(self.o))
        print(f"UNC = {UNC:.4f} (with {ob_unc=})")
        return UNC

    def compute_rel(self):
        """ Compute reliability.
        """
        rel_all = np.zeros_like(self.fk)
        for nk,k in enumerate(self.fk):
            ok = self.o[self.f==k]
            if ok.size != 0:
                ok_bar = self.do_mean(ok)
                rel_all[nk] = ok.size * ((k-ok_bar)**2)
        REL = np.sum(rel_all)/self.o.size
        print(f"{rel_all=}, REL = {REL:.4f}")
        return REL

    def compute_dsc(self):
        dsc_all = np.zeros_like(self.fk)
        o_bar = self.do_mean(self.o)
        for nk,k in enumerate(self.fk):
            ok = self.o[self.f==k]
            if ok.size != 0:
                ok_bar = self.do_mean(ok)
                dsc_all[nk] = ok.size * ((ok_bar-o_bar)**2)
        DSC = np.sum(dsc_all)/self.o.size
        print(f"{dsc_all=}, DSC = {DSC:.4f}")
        return DSC

    def compute_bss(self,from_components=False):
        if self.unc is not None:
            UNC = self.compute_UNC()
        else:
            UNC = self.unc
        BS = self.get_bs(from_components=from_components)
        BSS = (BS-UNC)/(0-UNC)
        return BSS
