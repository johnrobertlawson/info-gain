"""
Object-specific information gain. Uses the Divergence Score - tailored for object input

For the paper (Lawson et al 2020), we use 99th percentile CREF. Hence,
the score is for detection of the top 1% CREF, which is typically
associated in summer Great Plains with damaging hail and/or a strong thunderstorm.
"""

class OSIG:
    def __init__(self,fcst_array,obs_array,lats,lons,clip_probs=0.005):
        """
        For comp. refl, or one layer of refl. only right now.

        The two object_sets need to be in the structure:

        *(fcst)  cases x members x times x lats x lons
        *(obs)   cases x times x lats x lons       or
                 cases x 1 x times x lats x lons
        *(lats)  cases x lats x lats
        *(lons)  cases x lons x lons

        clip_probs - what to clip off the end of the forecast pdf to
            avoid divergence to inf.
        
        John Lawson, December 2019
        """
        assert fcst_array.ndims == 5

        if obs_array.ndims == 4:
            obs_array = obs_array[:,N.newaxis,:,:,:]
        else:
            assert obs_array.ndim = 5
        self.lats = lats
        self.lons = lons
            
        self.ncases, self.nmembers, self.= fcst_array.shape

        self.prob_bins = N.linspace(0,1,self.nmembers+1)
        self.K = len(self.prob_bins)

        fcst_object_set = self.identify_objects(fcst_array)
        obs_object_set = self.identify_object(obs_array)

        _ = self.calc_all(fcst_object_set,obs_object_set)
        self.DS, self.DSS, self.REL, self.RES, self.UNCS = _
        return

    def calc_REL(self,O_k,F_k):
        """
        O_k is frequency of an observed object matched with a k% object.
        #WRONG?# F_k is frequency of a forecast objects with a k% of occurrence
        F_k is % forecast
        n_k is the number of objects for this probability
        (N is the total number of objects)
        
        REL is relative entropy
        """
        kvector = N.zeros(self.prob_bins)
        for nk, k in enumerate(self.prob_bins):
            n_k = len()
            kvector[nk] = n_k * ((O_k * N.log2(O_k/F_k)) + (1-O_k * N.log_2((1-O_k)/(1-F_k))))
        REL = (1/self.N)*N.sum(kvector)
        return REL

    def calc_UNC(self,o):
        """
        o is the frequency of the object occurrence
        UNC is entropy
        """
        UNC = (-o * N.log2(o)) - ((1-o)*N.log2(1-o))
        return UNC

    def calc_RES(self,o_k,o):
        """
        o is the frequency of the object occurrence
        O_k is frequency of an observed object matched with a k% object.
        """
        kvector = N.zeros(K)
        for nk, k in enumerate(self.prob_bins):
            n_k = len()
            kvector[nk] = n_k * ((O_k * N.log2(O_k/F_k)) + (1-O_k * N.log_2((1-O_k)/(1-F_k))))

    def identify_objects(self,pc=99,dBZ=None,footprint=15):
        # data_masked = N.where(self.radmask,0.0,self.raw_data)
        obj_field_OK = N.zeros_like(data_masked)
        obj_init = N.where(data_masked >= self.threshold, data_masked, 0.0)
        obj_bool = obj_init > 0.0

        # 1, 2, 3...
        obj_labels = skimage.measure.label(obj_bool).astype(int)
        obj_labels_OK = N.zeros_like(obj_labels)

        # JRL: consider cache=False to lower memory but slow operation
        obj_props = skimage.measure.regionprops(obj_labels,obj_init)
        obj_OK = []
        for prop in obj_props:
            id = int(prop.label)
            size_OK = False
            not_edge_OK = False

            # Remove small storms
            if prop.area > (self.footprint/(self.dx**2)):
                size_OK = True

            # Remove storms touching perimeter
            # 1 km grid might have more storms that pass this check
            # This is 'fair', as that's the nature of the 1 km grid!
            # Also, a comparison might want to check if a storm was present
            # but removed - to avoid a false 'missed hit'. Instead, use a NaN.

            # print("Checking for an edge for object",id)
            # debug = True if id == 81 else False
            not_edge_OK = not self.check_for_edge(prop.coords,debug=False)

            # if all((size_OK,not_edge_OK)):
            if size_OK and not_edge_OK:
                obj_OK.append(id)

        # This replaces gridpoints within objects with original data
        for i,ol in enumerate(obj_OK):
            obj_field_OK = N.where(obj_labels==ol, obj_init, obj_field_OK)
            obj_labels_OK =  N.where(obj_labels==ol, obj_labels, obj_labels_OK)
        obj_props_OK = [obj_props[o-1] for o in obj_OK]
        # pdb.set_trace()
        return obj_field_OK, obj_props_OK, obj_OK, obj_labels, obj_labels_OK

    def list_of_edgecoords(self,pad=3):
        # TODO: consider doing outside 2 cells for 3km, and 6 cells for 1km
        pad = int(N.ceil(pad/self.dx))
        # These are all the values on the edge of the raw_data domain
        latidxs = N.indices(self.lats.shape)[0][:,0]
        lonidxs = N.indices(self.lats.shape)[1][0,:]

        ee0 = list(range(pad))
        elat0 = list(range(self.nlats-3,self.nlats))
        elon0 = list(range(self.nlons-3,self.nlons))
        elat = ee0 + elat0
        elon = ee0 + elon0

        _a = [(z,n) for n in lonidxs for z in elat]
        # _b = [(self.nlats,n) for n in lonidxs]
        _b = [(n,z) for n in latidxs for z in elon]
        # _d = [(n,self.nlons) for n in latidxs]

        edgecoords = _a + _b # + _c + _d
        # print("Edge coordinates have been calculated.")
        return edgecoords

    def check_for_edge(self,coords,debug=False):
        """ Returns True if any point is on the edge.
        """
        check_all = [(c[0],c[1]) in self.edgecoords for c in coords]
        if debug: pdb.set_trace()
        return any(check_all)

    def match_objects(self,t_max,d_max,cd_max):
        


        
