"""

TODO:
    * How to make utils (bounding) a pytest fixture?

John R. Lawson,  Valpo Uni 2022
"""
import logging
import copy

import pytest
import numpy as np

from utils.custom_exceptions import UnboundedProbError, SillyValueError

# https://stackoverflow.com/questions/4673373/logging-within-pytest-tests

LOGGER = logging.getLogger(__name__)
arg_names = "f,o"

# first, create some random sets, all same but with A-B testing
# random number generator with seed for repeatability
rng = np.random.default_rng(seed=27)

def create_longtail_timeseries(rng,n_fcsts=100):
    """Long tail, with (offset or error between forecasts
    and obs, how?)"""
    # Create a long tail for forecast of rare events
    mu, sigma = 3., 1. # mean and standard deviation
    raw_vals = rng.lognormal(mu, sigma, n_fcsts)
    return raw_vals

def create_unbounded_low(rng,raw_vals):
    # Unbounded, low values (very rare events)
    # leave as-is
    return copy.copy(raw_vals)

def create_unbounded_high(rng,raw_vals):
    # Unbounded, high values (very common events)
    return [1-n for n in ulvals]

def create_bounded_low(rng,raw_vals):
    return utils.bound(raw_vals,[0.02,0.98])
    # For uncertain obs, bound at 0.01 (1% uncertainty is max)

    # Randomised ob error versus consistent


# All permutations of forecast-observation pairs for testing of XES
# TODO: Both single pairs and a simulated time series

test_args = [
    # TODO: Work out the correct answer to each
    ########### SILLY VALUES ############
    (1.1,   0.4,    SillyValueError),       # Forecast val
    (0.4,   1.1,    SillyValueError),       # sim.
    (1.1,   1.2,    SillyValueError),       # sim.
    (-4.5,  0.5,    SillyValueError),       # sim.
    (0.5,   -4.5,   SillyValueError),       # sim.
    (0.5,   "NO!",  SillyValueError),       # sim.
    ########### SINGLE PAIRS ############
    (1,     0,      UnboundedProbError),     # Passing integers
    (0,     1,      UnboundedProbError),      # sim.
    (1,     1,      UnboundedProbError),      # sim.
    (0,     0,      UnboundedProbError),      # sim.
    (0.4,   1.0,    UnboundedProbError),       # Bounded fcst; binary obs
    (0.4,   0.0,    UnboundedProbError),    # sim.
    (1.0,   0.0,    UnboundedProbError),    # Unbounded; binary obs
    (1.0,   1.0,    UnboundedProbError),    # sim.
    (0.0,   0.0,    UnboundedProbError),    # sim.
    (0.0,   1.0,    UnboundedProbError),    # sim.
    (0.4,   0.99,   5.5555),    # Bounded; unc obs
    (0.4,   0.01,   5.5555),     # sim.
    (1.0,   0.01,   5.5555),     # Unbounded; unc obs
    (1.0,   0.99,   5.5555),     # sim.
    (0.0,   0.01,   5.5555),     # sim.
    (0.0,   0.99,   5.5555),     # sim.
    (0.0,   0.0,    5.5555),     # Same values together
    (0.02,  0.02,   5.5555),     # sim.
    (0.98,  0.98,   5.5555),   # sim.
    (1.0,   1.0,    5.5555), # sim.
    (0.02,  0.01,   5.5555), # More error in fcst
    (0.01,  0.02,   5.5555), # More error in obs (valid?!)
    (0.02,  0.0,    5.5555),  # No error in obs (close to certainty)
    (0.6,   1.0,    5.5555),    # sim.
    ########### FAKE TIME SERIES ############
    (f_1,   o_1,    5.5555),    #
    ]



# Template for parametrization settings
# @pytest.mark.parametrize(
#     "param1,param2",
#     [
#         ("a", "b"),
#         ("c", "d"),
#     ],
# )

@pymark.mark.parametrize(arg_names,test_args)
class TestCrossEntropy:
    @pytest.fixture()
    def setup(f,o):
        XES = CrossEntropy(f,o)
        return XES

    @pytest.mark.parametrize("f,o", test_args)
    def test_one_fcst_xes(self,f,o):
        XES = CrossEntropy(f,o)
        xes = XES.compute_XES()
        LOGGER.info(f"Using {f=:.3f} and {o=.3f} gives {xes}.")
        assert isinstance(xes,float)
        return

