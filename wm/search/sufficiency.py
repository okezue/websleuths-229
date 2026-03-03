from __future__ import annotations
from wm.types import Claim,Community

def is_sufficient(claims:list[Claim],communities:list[Community],
                  prev_n:int=0,min_claims:int=5)->bool:
    n=len(claims)
    if n<min_claims:return False
    if prev_n>0:
        gain=(n-prev_n)/max(prev_n,1)
        if gain<0.1:return True
    if communities:
        for co in communities:
            mc=sum(1 for c in claims if c.cid in co.members)
            if mc<2:return False
        return True
    return n>=min_claims
