import pdb
import os

import numpy as N
import matplotlib as M
import matplotlib.pyplot as plt

def farey_operator(IC,nt=75):
    ts = N.zeros([nt])
    ts[0] = IC
    for n in range(nt-1):
        if 0 <= ts[n] <= 0.5:
            y = ts[n]/(1-ts[n])
        elif 0.5 < ts[n] <= 1:
            y = (1-ts[n])/ts[n]
        else:
            raise Exception
        ts[n+1] = y
    return ts

root = 4
fig,axes = plt.subplots(root,root,figsize=(16,12))
fname = "farey_test.png"
fpath = fname

n_ic = root**2

ICs_truth = N.zeros(n_ic)
for n in range(n_ic):
    # minmax = [0,1]
    # mm = [0.8165,0.8175]
    mm = [0.3333,0.3334]
    ICs_truth[n] = N.random.uniform(min(mm),max(mm))

for ax, ic in zip(axes.flat,ICs_truth):
    ts_truth = farey_operator(ic)
    # print(ts_truth)
    ax.plot(ts_truth)
    ax.set_title(f"{ic:.3f}")

fig.tight_layout()
fig.savefig(fpath)

