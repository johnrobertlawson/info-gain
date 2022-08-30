"""Run parallel logic to the paper.

Similar to Jupyter notebook but without the need for jupyter. And much easier
to develop on, IMO!

JRL, Valparaiso Univ., Autumn 2022
"""

### Run examples for single numbers on the XES and BS that
# (1) tests the scripts and
# (2) demonstrates the difference in interpretation, maybe.
# We can generate differences in XESS and BSS by setting an UNC (1, 10, 30%?)

### Run Lorenz-63 intermittent model with "truth" and "forecast"
# Verify the performance of various factors that mimic a windowing
# verification like eFSS. Use both XESS and BSS, broken down by
# component, to highlight the differences in interpretation.

# (1) Generate a time series of truth
# (2) Generate 500 ensemble members perturbed from (1) using different factors
# (3) Windowing at 1,3,5 etc., ready to verify a bounded pc from ensemble
# (4)
