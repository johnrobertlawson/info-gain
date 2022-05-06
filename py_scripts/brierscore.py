import pdb

import numpy as np
import pandas as pd

np.set_printoptions(precision=4,suppress=True)
pd.set_option("display.precision",4)
pd.set_option("display.float_format",'{:.4f}'.format)

class BrierScore:
    def __init__(self,f,o,fk=None,ob_uncertainty=True):
        self.o = np.array(o)
        self.f = np.array(f)

        if fk is None:
            self.fk = np.unique(self.f)
        # NOT IMPLEMENTED
        else:
            pdb.set_trace()
            # for k in np.unique(self.f):
            #     assert k in fk
            # self.fk = np.array(fk)

        self.ob_uncertainty = ob_uncertainty

    @staticmethod
    def do_mean(x):
        if isinstance(x,(float,int)):
            return x
        else:
            return np.mean(x)

    def compute_BS(self,from_components=False):
        if from_components:
            REL = self.compute_REL()
            DSC = self.compute_DSC()
            UNC = self.compute_UNC()
            return REL - DSC + UNC
        else:
            return np.sum((self.f-self.o)**2)/self.o.size

    def compute_UNC(self):
        if self.ob_uncertainty:
            UNC = np.sum((self.o-self.do_mean(self.o))**2)/self.o.size
        else:
            UNC = self.do_mean(self.o) * (1-self.do_mean(self.o))
        print(f"UNC = {UNC:.4f} (with {self.ob_uncertainty=})")
        return UNC

    def compute_REL(self):
        """ Compute reliability.
        """
        rel = np.zeros_like(self.fk)
        for nk,k in enumerate(self.fk):
            ok = self.o[self.f==k]
            if ok.size != 0:
                ok_bar = self.do_mean(ok)
                rel[nk] = ok.size * ((k-ok_bar)**2)
        REL = np.sum(rel)/self.o.size
        print(f"{rel=}, REL = {REL:.4f}")
        return REL

    def compute_DSC(self):
        dsc = np.zeros_like(self.fk)
        o_bar = self.do_mean(self.o)
        for nk,k in enumerate(self.fk):
            ok = self.o[self.f==k]
            if ok.size != 0:
                ok_bar = self.do_mean(ok)
                dsc[nk] = ok.size * ((ok_bar-o_bar)**2)
        DSC = np.sum(dsc)/self.o.size
        print(f"{dsc=}, DSC = {DSC:.4f}")
        return DSC

    def compute_BSS(self,from_components=False):
        BS_UNC = self.compute_UNC()
        BS = self.compute_BS(from_components=from_components)
        BSS = (BS-BS_UNC)/(0-BS_UNC)
        return BSS
