from __future__ import annotations
import re
from wm.types import Chunk

def _sent_split(txt:str)->list[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+',txt) if s.strip()]

class CitedQaRecipe:
    def __init__(self,max_len:int=1024):
        self._ml=max_len
    def build(self,chunks:list[Chunk])->list[dict]:
        rows=[]
        for c in chunks:
            sents=_sent_split(c.text)
            if len(sents)<3:continue
            ctx=" ".join(sents)
            q=f"What does this passage say about {sents[0][:60].strip('.!?')}?"
            a=sents[1]
            rows.append({
                "prompt":q,"completion":f"{a} [source: {c.eid}]",
                "context":ctx[:self._ml*4],"eid":c.eid,
            })
        return rows
