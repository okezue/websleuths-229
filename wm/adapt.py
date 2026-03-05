from __future__ import annotations
import math
from wm.types import SearchResult

def adaptive_steps(sr:SearchResult,lo:int=30,hi:int=150,
                   a:float=0.5,b:float=10.0)->int:
    n_claims=len(sr.claims)
    mean_conf=sum(c.confidence for c in sr.claims)/max(n_claims,1)
    n_comm=len(sr.communities)
    raw=a*mean_conf*math.log(1+n_claims)+b*math.log(1+n_comm)
    t=1.0/(1.0+math.exp(-(raw-3.0)))
    return int(max(lo,min(hi,lo+t*(hi-lo))))

def round_temp(temps:list[float],rnd:int)->float:
    if not temps:return 0.9
    return temps[min(rnd,len(temps)-1)]
