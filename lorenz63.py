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

class Lorenz63:
    def __init__(self,):
        """
        Args:
            * clip_first (int): remove first n steps to avoid spin-up
        """
        self.clip_first = 0

    def do_lorenz63(self,):
        pass

    def generate_raw(self,):
        pass

    def clip(self,):
        # Removes self.clip_first timesteps to avoid spin-up
        pass

    def generate_perturbated(self,):
        pass 
