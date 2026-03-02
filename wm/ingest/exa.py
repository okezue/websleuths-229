from __future__ import annotations
import os
from datetime import datetime
from wm.types import Episode

def _get_exa_cls():
    from exa_py import Exa
    return Exa

class ExaSrc:
    def __init__(self,api_key:str|None=None):
        self._key=api_key or os.environ.get("EXA_API_KEY","")
    def fetch(self,query:str,n:int=10)->list[Episode]:
        if not self._key:
            raise RuntimeError("EXA_API_KEY not set")
        try:
            Exa=_get_exa_cls()
        except ImportError:
            raise RuntimeError("exa-py not installed")
        c=Exa(self._key)
        res=c.search_and_contents(query,num_results=n,text=True)
        eps=[]
        for r in res.results:
            sc=getattr(r,"score",0.0) or 0.0
            eps.append(Episode(
                url=r.url,title=r.title or "",
                body=r.text or "",
                ts=datetime.utcnow(),
                authority=max(0.0,min(1.0,sc/max(abs(sc),1e-8))) if sc else 0.5,
                meta={"source":"exa","raw_score":sc},
            ))
        return eps
