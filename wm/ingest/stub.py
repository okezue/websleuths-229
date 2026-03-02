from __future__ import annotations
from datetime import datetime
from wm.types import Episode

class StubSrc:
    def __init__(self,n_default:int=5):
        self._n=n_default
    def fetch(self,query:str,n:int=0)->list[Episode]:
        k=n or self._n
        eps=[]
        for i in range(k):
            eps.append(Episode(
                url=f"https://stub.test/{query}/{i}",
                title=f"Stub {query} #{i}",
                body=f"Body of stub episode {i} about {query}. "*10,
                ts=datetime(2025,1,1+i%28),
                authority=0.5+0.05*i,
            ))
        return eps
