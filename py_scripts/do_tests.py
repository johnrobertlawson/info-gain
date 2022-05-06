import pdb

import numpy as np
import pandas as pd

from brierscore import BrierScore
from crossentropy import CrossEntropy

np.set_printoptions(precision=4,suppress=True)
pd.set_option("display.precision",4)
pd.set_option("display.float_format",'{:.4f}'.format)

def print_tests(f,o,fk,title,ob_unc):
    fk = None
    brier = BrierScore(f,o,fk=fk,ob_uncertainty=ob_unc)

    print(title,(" "*5),f"BS = {brier.compute_BS():.4f}",("="*20),)
    print(f"BS from comp = {brier.compute_BS(from_components=True):.4f}\n")
    # print("\n")
    return

# Testing for one forecast

# Probs are unbounded
# Obs are binary
f = [0.4,]
o = [1,]
fk = np.arange(0,1.01,0.1)
print_tests(f,o,fk,"SINGLE:",False)

###########################################################

# Testing for multiple forecasts

# Probs are unbounded
# Obs are binary
f = [0.1,0.5,0.9,0.8,0.7,0.2,1,0,0.1]
o = [0,0,1,1,0,0,1,1,0]
fk = np.arange(0,1.01,0.1)
print_tests(f,o,fk,"RAW FALSE:",False)

# Probs are bounded to [0.01,0.99]
# Obs are binary
f = [0.1,0.5,0.9,0.8,0.7,0.2,0.99,0.01,0.1]
o = [0,0,1,1,0,0,1,1,0]
fk = [0.01,] + list(np.arange(0.1,0.91,0.1)) + [0.99,]
print_tests(f,o,fk,"PROBS 1pc:",False)

# Probs are bounded to [0.01,0.99]
# Obs are binary
f = [0.1,0.5,0.9,0.8,0.7,0.2,0.99,0.01,0.1]
o = [0,0,1,1,0,0,1,1,0]
print_tests(f,o,None,"PROBS 1pc -- NO FK:",False)

# Probs are bounded to [0.01,0.99]
# Obs are from {0.01,0.99} (same certainty)
f = [0.1,0.5,0.9,0.8,0.7,0.2,0.99,0.01,0.1]
o = [0.01,0.01,0.99,0.99,0.01,0.01,0.99,0.99,0.01]
fk = [0.01,] + list(np.arange(0.1,0.91,0.1)) + [0.99,]
print_tests(f,o,fk,"BOTH 1pc:",True)

# Probs are bounded to [0.01,0.99]
# Obs are from {0.001,0.999} (high certainty)
f = [0.1,0.5,0.9,0.8,0.7,0.2,0.99,0.01,0.1]
o = [0.001,0.001,0.999,0.999,0.001,0.001,0.999,0.999,0.001]
fk = [0.01,] + list(np.arange(0.1,0.91,0.1)) + [0.99,]
print_tests(f,o,fk,"OBS 0.1pc:",True)

# Testing for multiple forecasts
# Probs are bounded to [0.01,0.99]
# Obs are from {0,0.999} (ignore null forecast error)
f = [0.1,0.5,0.9,0.8,0.7,0.2,0.99,0.01,0.1]
o = [0,0,0.999,0.999,0,0,0.999,0.999,0]
fk = [0.01,] + list(np.arange(0.1,0.91,0.1)) + [0.99,]
print_tests(f,o,fk,"IGNORE NULL",True)
