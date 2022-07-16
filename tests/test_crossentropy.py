"""

TODO:
    * How to make utils (bounding) a pytest fixture?

John R. Lawson,  Valpo Uni 2022
"""
import logging
import pytest
import copy

import numpy as np

# https://stackoverflow.com/questions/4673373/logging-within-pytest-tests
LOGGER = logging.getLogger(__name__)

class TestCrossEntropy:
    fo_pairs = [
                (1,0),      # Passing integers
                (0,1),      # sim.
                (1,1),      # sim.
                (0,0),      # sim.
                (0.4,1.0),  # Bounded fcst; binary obs
                (0.4,0.0),  # sim.
                (1.0,0.0),  # Unbounded; binary obs
                (1.0,1.0),  # sim.
                (0.0,0.0),  # sim.
                (0.0,1.0),  # sim.
                (0.4,0.99), # Bounded; unc obs
                (0.4,0.01), # sim.
                (1.0,0.01), # Unbounded; unc obs
                (1.0,0.99), # sim.
                (0.0,0.01), # sim.
                (0.0,0.99), # sim.
                (0.0,0.0),  # Same values together
                (0.02,0.02),# sim.
                (0.98,0.98),# sim.
                (1.0,1.0),  # sim.
                (0.02,0.01),# More error in fcst
                (0.01,0.02) # More error in obs (valid?!)
                (0.02,0.0)  # No error in obs (close to certainty)
                (0.6,1.0)   # sim.
            ]

    @pytest.fixture()
    def setup(f,o):
        XES = CrossEntropy(f,o)
        return XES

    @pytest.mark.parametrize("f,o",fo_pairs)
    def test_one_fcst_xes(self,f,o):
        XES = CrossEntropy(f,o)
        xes = XES.compute_XES()
        LOGGER.info(f"Using {f=} and {o=} gives {xes}.")
        assert isinstance(xes,float)
        return

    def test_one_fcst_bounded_binaryobs(self):
        f = 0.4
        o = 1.0

    def test_one_fcst_unbounded_binaryobs(self):
        f = 1.0
        o = 0.0

    def test_many_fcsts_permutations(self,n_fcsts=100):
        # first, create some random sets, all same but with A-B testing
        # random number generator with seed for repeatability
        rng = np.random.default_rng(seed=27)

        # Create a long tail for forecast of rare events
        mu, sigma = 3., 1. # mean and standard deviation
        vals = rng.lognormal(mu, sigma, n_fcsts)

        # Unbounded, low values
        # leave as-is
        ulvals = copy.copy(vals)

        # Unbounded, high values
        uhvals = [1-n for n in ulvals]

        # Bounded, low values 
        # Bound at 0.02 (2% is lower prob that is issue)



        # For uncertain obs, bound at 0.01 (1% uncertainty is max)


        # Randomised ob error versus consistent


        # Now create the test permutations
        fo_pairs = []

        return

    def silly_value_perms(self):
        """Do permutations of nonsensical values
        """
        all_silly_vals = [
                        (1.1,0.1),
                        ]
        return

