from __future__ import annotations
import math,numpy as np

def paired_bootstrap_ci(pre:list[float],post:list[float],
                        n_boot:int=1000,alpha:float=0.05)->tuple[float,float]:
    pre_a,post_a=np.array(pre),np.array(post)
    diffs=post_a-pre_a
    n=len(diffs)
    rng=np.random.default_rng(42)
    means=np.empty(n_boot)
    for i in range(n_boot):
        idx=rng.integers(0,n,size=n)
        means[i]=diffs[idx].mean()
    lo=np.percentile(means,100*alpha/2)
    hi=np.percentile(means,100*(1-alpha/2))
    return float(lo),float(hi)

def sprt_test(pre:list[float],post:list[float],
              delta:float=0.05,alpha:float=0.05,beta:float=0.1)->str:
    A=math.log((1-beta)/alpha)
    B=math.log(beta/(1-alpha))
    llr=0.0
    for p0,p1 in zip(pre,post):
        d=p1-p0
        llr+=d/max(delta,1e-8)
        if llr>=A:return "accept"
        if llr<=B:return "reject"
    return "undecided"
