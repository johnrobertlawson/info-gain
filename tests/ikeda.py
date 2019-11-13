import pdb
import os

import numpy as N
import matplotlib as M
import matplotlib.pyplot as plt

def ikeda_operator(x0=0.48,y0=0.47,p=7.275,A=0.85,B=0.9,k=0.4,nt=70):
    x_ts = N.zeros([nt])
    y_ts = N.zeros([nt])
    x_ts[0] = x0
    y_ts[0] = y0

    for n in range(nt-1):
        wn = 1 + x_ts[n]**2 + y_ts[n]**2
        x_ts[n+1] = A + B*x_ts[n]*N.cos(k - (p/wn)) - B*y_ts[n]*N.sin(k - (p/wn))
        y_ts[n+1] = B*y_ts[n]*N.cos(k - (p/wn)) + B*x_ts[n]*N.sin(k - (p/wn))
    return x_ts, y_ts

root = 4
fig,axes = plt.subplots(root,root,figsize=(16,12))
fname = "ikeda_test.png"
fpath = fname

n_ic = root**2

x0s = N.zeros(n_ic)
y0s = N.zeros(n_ic)
psets = N.zeros(n_ic)

for n in range(n_ic):
    # mm = [0,1]
    mm = [0.9,]
    # mm = [0.8165,0.8175]
    # mm = [0.3333,0.3334]
    x0s[n] = N.random.uniform(min(mm),max(mm))
    y0s[n] = N.random.uniform(min(mm),max(mm))

    # mm2 = [7.27,7.28]
    optimal_p = 7.26884894
    optimal_p += 0.01
    mm2 = [optimal_p - 1E-5, optimal_p + 1E-5]
    psets[n] = N.random.uniform(min(mm2),max(mm2))
    # psets[n] = optimal_p

for ax, x0, y0, pset in zip(axes.flat,x0s,y0s,psets):
    x_ts, y_ts = ikeda_operator(x0=x0,y0=y0,p=pset)
    x_ts[x_ts < 0] = 0
    y_ts[y_ts < 0] = 0
    # print(ts_truth)
    ax.plot(y_ts)
    ax.set_title(f"p = {pset:.3f} || x0 = {x0:.3f} || y0 = {y0:.3f}",fontsize=8)

fig.tight_layout()
fig.savefig(fpath)

