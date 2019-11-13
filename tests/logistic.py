import pdb
import os

import numpy as N
import matplotlib as M
M.use("agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy as S

# mu = 3.83
def logistic_operator(IC,nt=30,mu=3.81):
    # values for mu
    mm = [3.56995,3.82843]
    mu = N.random.uniform(min(mm),max(mm))
    ts = N.zeros([nt])
    ts[0] = IC
    for n in range(nt-1):
        tiny = 0.1
        mu += N.random.uniform(-tiny,tiny)
        ts[n+1] = mu*ts[n]*(1-ts[n])
    return ts, mu

root = 4
fig,axes = plt.subplots(root,root,figsize=(16,12))
fname = "logistic_test.png"
fpath = fname

fig2,axes2 = plt.subplots(root,root,figsize=(16,12))
fname2 = "logistic_test_alt.png"
fpath2 = fname2

n_ic = root**2

ICs_truth = N.zeros(n_ic)
mus = []

for n in range(n_ic):
    # minmax = [0,1]
    # mm = [0.8165,0.8175]
    mm = [0.001,0.999]
    # mm = [0.754,0.756]
    ICs_truth[n] = N.random.uniform(min(mm),max(mm))
    # mus.append(mu)

for ax, ax2, ic in zip(axes.flat,axes2.flat,ICs_truth):
    ts_truth, mu = logistic_operator(ic)
    # top_pc = N.quantile(ts_truth,0.995)
    # ts_truth[ts_truth<top_pc] = 0

    # print(ts_truth)
    ax.plot(ts_truth)
    ax.set_title(f"IC = {ic:.3f}  ||  mu = {mu:.3f}")

    ts_truth_disc = N.clip(ts_truth,0.8,0.9)
    # autocorr = sm.tsa.acf(ts_truth, nlags=10)
    ax2.plot(ts_truth_disc)
    ax.set_title(f"IC = {ic:.3f}  ||  mu = {mu:.3f}")
    # pdb.set_trace()


fig.tight_layout()
fig.savefig(fpath)
fig2.tight_layout()
fig2.savefig(fpath2)

