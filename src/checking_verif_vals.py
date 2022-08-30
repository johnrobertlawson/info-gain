import pdb

import numpy as np
import pandas as pd

from brierscore import BrierScore
from crossentropy import CrossEntropy

np.set_printoptions(precision=4,suppress=True)
pd.set_option("display.precision",4)
pd.set_option("display.float_format",'{:.4f}'.format)

do_brier = False
do_xes = True
def print_tests(f,o,title,ob_unc):
    if do_brier:
        brier = BrierScore(f,o,ob_uncertainty=ob_unc)
        print(title,(" "*5),f"BS = {brier.compute_BS():.4f}",("="*20),)
        print(f"BS from comp = {brier.compute_BS(from_components=True):.4f}\n")
    if do_xes:
        xes = CrossEntropy(f,o)
        print(title,(" "*5),
            f"XES = {xes.compute_XES(from_components=False):.4f}",
                        ("="*20),)
        print(f"XES from comp= {xes.compute_XES(from_components=True):.4f}\n")
    return


# Testing for one forecast

# Probs are unbounded
# Obs are binary
f = [0.4,]
o = [1,]
print_tests(f,o,"SINGLE:",False)

###########################################################

# Testing for multiple forecasts

# Probs are unbounded
# Obs are binary
f = [0.1,0.5,0.9,0.8,0.7,0.2,1,0,0.1]
o = [0,0,1,1,0,0,1,1,0]
print_tests(f,o,"RAW:",False)

# Probs are bounded to [0.01,0.99]
# Obs are binary
f = [0.1,0.5,0.9,0.8,0.7,0.2,0.99,0.01,0.1]
o = [0,0,1,1,0,0,1,1,0]
print_tests(f,o,"PROBS 1pc:",False)

# Probs are bounded to [0.01,0.99]
# Obs are from {0.01,0.99} (same certainty)
f = [0.1,0.5,0.9,0.8,0.7,0.2,0.99,0.01,0.1]
o = [0.01,0.01,0.99,0.99,0.01,0.01,0.99,0.99,0.01]
print_tests(f,o,"BOTH 1pc FALSE:",False)

# Probs are bounded to [0.01,0.99]
# Obs are from {0.001,0.999} (high certainty)
f = [0.1,0.5,0.9,0.8,0.7,0.2,0.99,0.01,0.1]
o = [0.001,0.001,0.999,0.999,0.001,0.001,0.999,0.999,0.001]
print_tests(f,o,"OBS 0.1pc:",True)

# Testing for multiple forecasts
# Probs are bounded to [0.01,0.99]
# Obs are from {0,0.999} (ignore null forecast error)
f = [0.1,0.5,0.9,0.8,0.7,0.2,0.99,0.01,0.1]
o = [0,0,0.999,0.999,0,0,0.999,0.999,0]
print_tests(f,o,"IGNORE NULL",True)
