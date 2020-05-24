"""Verification for info-gain datasets: idealised/Lorenz63 only for time being.

Requirements:
    * Scale-aware information gain (IG_sa)
    * Ensemble Fractions Skill Score (eFSS)
        - Also Brier Score (BS)?
    * Need to decompose the above - with/without obs error included
        - This is cross-entropy v D_KL for info gain


Experiments/verification (permutation of some below, e.g. with/without tolerance
        for FSS v IG_sa):
    * Compare time-tolerance vs. no tolerance
    * Compare FSS v IGsa (does this reduce to BS v IGN?)
    * Compare cross-entropy (incl. obs error) vs D_KL
        - This may need checking errors in time v errors in magnitude
        - Check sensitivity to magnitude of observational error
    * Look at decomposed IGsa (and maybe decompose FSS/BS?)
        - Check cross-entropy vs. D_KL again
    * Compare range of regime intermittency (more/less rare)
        - Show spectrum to show long tails - before/after event filtering?
        - Does FSS/BS or IGsa reward rare forecasts better?
    * Compare filtering of events v no filtering (i.e., time windowing performed
        on the continuous data)
"""

class Verif:
    def __init__(self,):
        """
        Args:
            * What
        """
        pass

    def IGsa(self):
        pass

    def eFSS(self):
        pass
