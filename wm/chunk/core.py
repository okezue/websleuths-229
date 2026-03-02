from __future__ import annotations
import re
import tiktoken
from wm.types import Episode,Chunk
from wm.cfg import ChunkCfg

def _norm_md(txt:str)->str:
    txt=re.sub(r'\n{3,}','\n\n',txt)
    txt=re.sub(r'[ \t]+',' ',txt)
    return txt.strip()

class Chunker:
    def __init__(self,cfg:ChunkCfg):
        self._cfg=cfg
        self._enc=tiktoken.get_encoding(cfg.enc)
    def chunk(self,ep:Episode)->list[Chunk]:
        txt=_norm_md(ep.body)
        toks=self._enc.encode(txt)
        if not toks:return []
        chunks=[]
        step=max(1,self._cfg.max_tok-self._cfg.overlap)
        i=0
        idx=0
        while i<len(toks):
            window=toks[i:i+self._cfg.max_tok]
            chunks.append(Chunk(
                eid=ep.eid,idx=idx,
                text=self._enc.decode(window),
                n_tok=len(window),
                authority=ep.authority,
            ))
            idx+=1
            i+=step
        return chunks
    def chunk_many(self,eps:list[Episode])->list[Chunk]:
        out=[]
        for e in eps:
            out.extend(self.chunk(e))
        return out
