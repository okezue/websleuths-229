from __future__ import annotations
import re
from wm.types import Chunk

def _sent_split(txt:str)->list[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+',txt) if s.strip()]

class ExtSftRecipe:
    def __init__(self,max_len:int=1024):
        self._ml=max_len
    def build(self,chunks:list[Chunk])->list[dict]:
        rows=[]
        for c in chunks:
            sents=_sent_split(c.text)
            if len(sents)<2:continue
            ctx=" ".join(sents[:-1])
            ans=sents[-1]
            rows.append({
                "prompt":f"Extract the key fact from:\n{ctx[:self._ml*4]}",
                "completion":ans,"eid":c.eid,
            })
        return rows
