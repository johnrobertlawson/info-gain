"""
Divergence Score - tailored for object input

For the paper (Lawson et al 2020), we use 99th percentile CREF. Hence,
the score is for detection of the top 1% CREF, which is typically
associated in summer Great Plains with damaging hail and/or a strong thunderstorm.
"""

# class OSIG:
class DivergenceScore:
    def __init__(self,fcst_array,obs_array):
        """
        For comp. refl, or one layer of refl. only right now.

        The two object_sets need to be in the structure:

        *   (fcst)  cases x members x times x lats x lons
        *   (obs)   cases x times x lats x lons       or
                    cases x 1 x times x lats x lons

        John Lawson, December 2019
        """
        if obs_array.ndims = 3:
            obs_array = obs_array[:,N.newaxis,:,:,:]
            
        self.prob_bins = 0
        self.N = 0
        self.K = len(self.prob_bins)

        fcst_object_set = self.identify_objects(fcst_array)

        _ = self.calc_all(fcst_object_set,obs_object_set)
        self.DS, self.DSS, self.REL, self.RES, self.UNCS = _
        self.DS
        self.DSS = DSS

    def calc_REL(self,O_k,F_k):
        """
        O_k is frequency of an observed object matched with a k% object.
        F_k is frequency of a forecast objects with a k% of occurrence
        n_k is the number of objects for this probability
        (N is the total number of objects)
        """
        kvector = N.zeros(K)
        for nk, k in enumerate(self.prob_bins):
            n_k = len()
            kvector[nk] = n_k * ((O_k * N.log2(O_k/F_k)) + (1-O_k * N.log_2((1-O_k)/(1-F_k))))
        REL = (1/self.N)*N.sum(kvector)
        return REL

    def identify_objects(self,data,pc=99,dBZ=None,footprint=15):
        pass

    def match_objects(self,t_max,d_max,cd_max):
        pass


        
