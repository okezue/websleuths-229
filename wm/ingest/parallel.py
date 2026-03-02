from __future__ import annotations
from wm.types import Episode

class ParallelSrc:
    def __init__(self,srcs:list):
        self._srcs=srcs
    def fetch(self,query:str,n:int=10)->list[Episode]:
        seen=set()
        eps=[]
        for s in self._srcs:
            for e in s.fetch(query,n):
                if e.eid not in seen:
                    seen.add(e.eid)
                    eps.append(e)
        return eps[:n]
