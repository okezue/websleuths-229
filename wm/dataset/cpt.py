from __future__ import annotations
from wm.types import Chunk

class CptRecipe:
    def __init__(self,max_len:int=1024):
        self._ml=max_len
    def build(self,chunks:list[Chunk])->list[dict]:
        rows=[]
        for c in chunks:
            rows.append({"text":c.text[:self._ml*4],"eid":c.eid,"idx":c.idx,
                "authority":c.authority})
        return rows
