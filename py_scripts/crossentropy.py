class CrossEntropy:
    """ For computing and visualising skill scores for a probabilistic forecast.
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
        self.f = f
        self.o = o
        self.logbase = 2

        if fk == None:
            self.fk = np.unique(self.f)
        else:
            self.fk = fk

    @staticmethod
    def do_geomean(n):
        # if isinstance(n,(np.float,float,np.int,int)):
        # pdb.set_trace()
        if isinstance(n,(float,int)):
            return n
        else:
            return geomean(n)

    @staticmethod
    def compute_DKL(x,y,return_all=False):
        """ Kullback-Liebler Divergence

        Args:
        x   : 1-D (e.g., observations)
        y   : 1-D (e.g., forecasts)
        """
        with np.errstate(divide='ignore',invalid='ignore'):
            # Enforce array so we can remove nans
            term1 = np.array(((1-x) * np.log2((1-x)/(1-y))))
            term2 = np.array((x * np.log2(x/y)))

        term1[x==1] = 0
        term2[x==0] = 0

        DKL_all = term1 + term2
        DKL = np.mean(DKL_all)
        if return_all:
            return DKL, DKL_all
        return DKL

    def compute_XESS(self,with_H=True,from_components=False):
        UNC = self.compute_UNC()
        XES = self.compute_XES(with_H=with_H)
        XESS = (XES-UNC)/(0-UNC)
        return XESS

    @staticmethod
    def compute_entropy(x,return_all=False):
        """ Average surprise. A series of surprise
        values are returned if return_all is True.

        Is this right? Surprise v UNC?

        """
        H_all = (-(1-x) * np.log2(1-x)) + -(x * np.log2(x))
        # H_all = ((1-x) * np.log2(1-x)) - (x * np.log2(x))
        H = np.mean(H_all)
        if return_all:
            return H, H_all
        return H

    def compute_H(self):
        """Return entropy of the observation field.
        """
        return self.compute_entropy(self.o)

    def compute_XES(self,with_H=True,from_components=False,return_all=False):
        """ Cross-entropy score.

        Args:

        """
        if with_H:
            if return_all:
                H, H_all = self.compute_H(return_all=True)
            else:
                H = self.compute_H()

        if from_components:
            REL = self.compute_REL()
            DSC = self.compute_DSC()
            UNC = self.compute_UNC()
            if with_H:
                XES = REL-DSC+UNC+H
            else:
                XES = REL-DSC+UNC
        else:
            if return_all:
                XES, XES_all = self.compute_DKL(self.o,self.f,return_all=True)
            else:
                XES = self.compute_DKL(self.o,self.f)

            if with_H:
                XES += H
                if return_all:
                    XES_all += H_all
                    return XES, XES_all

        return XES

    @staticmethod
    def single_XES(f,o):
        """No observational error!
        """
        return ((1-o)*np.log2((1-o)/(1-f)))+(o*np.log2(o/f))

    @classmethod
    def single_XESS(cls,f,o):
        """No observational error!
        """
        XES = cls.single_XES(f,o)
        XES_UNC = cls.single_UNC(f,o)
        XESS = (XES-XES_UNC)/(0-XES_UNC)
        return XESS

    @staticmethod
    def single_UNC(f,o):
        """No observational error!
        """
        return (-o*np.log2(o))-((1-o)*np.log2(1-o))

    def compute_DSC(self):
        dsc = np.zeros_like(self.fk)
        for nk,k in enumerate(self.fk):
            ok = self.o[self.f==k]
            ok_hat = self.do_geomean(ok)
            o_hat = self.do_geomean(self.o)
            dsc[nk] = ok.size * self.compute_DKL(ok_hat,o_hat)
        DSC = np.sum(dsc)/self.o.size
        return DSC

    def compute_REL(self):
        rel = np.zeros_like(self.fk)
        for nk,k in enumerate(self.fk):
            ok = self.o[self.f==k]
            ok_hat = self.do_geomean(ok)
            rel[nk] = ok.size * self.compute_DKL(ok_hat,k)
        REL = np.sum(rel)/self.o.size
        return REL

    def compute_UNC(self):
        """ Compute uncertainty component of the forecast.
        """
        p_o = self.do_geomean(self.o)
        # with np.errstate(divide='ignore',invalid='ignore'):
        term1 = -p_o*np.log2(p_o)
        term2 = (1-p_o)*np.log2(1-p_o)
        UNC = term1 - term2
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

    def count_bin_size(x,fk):
        """From Wilks: need to bin fcst probs, setting the middle of
        the bin as the value...

        """
        bin_counts,bin_edges = np.histogram(x,bins=fk)
        return bin_counts

    def quantise(x,fk):
        # I think the index is not right, need to look at bin edges?
        quant_probs = np.digitize(x,fk)
        # pdb.set_trace()
        f = [fk[q-1] for q in quant_probs]
        return f
