from __future__ import annotations
from wm.types import Episode

def exa_authority(eps:list[Episode])->list[Episode]:
    if not eps:return eps
    mx=max((e.authority for e in eps),default=1.0) or 1.0
    for e in eps:
        e.authority=e.authority/mx
    eps.sort(key=lambda e:e.authority,reverse=True)
    return eps

pagerank_authority=exa_authority
