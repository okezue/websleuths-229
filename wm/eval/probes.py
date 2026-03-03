from __future__ import annotations
import re,hashlib
from wm.types import Probe,Chunk

def _sent_split(txt:str)->list[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+',txt) if s.strip()]

def _mask_entities(s:str)->tuple[str,str]|None:
    m=re.search(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|\d[\d,.]+)\b',s)
    if not m:return None
    masked=s[:m.start()]+"____"+s[m.end():]
    return masked,m.group()

class ProbeBuilder:
    def build_cloze(self,chunks:list[Chunk])->list[Probe]:
        probes=[]
        for c in chunks:
            for s in _sent_split(c.text):
                r=_mask_entities(s)
                if r is None:continue
                masked,gold=r
                qid=hashlib.sha256(masked.encode()).hexdigest()[:12]
                probes.append(Probe(
                    qid=qid,prompt=f"Fill in the blank: {masked}",
                    gold=gold,eid=c.eid,kind="cloze",
                ))
        return probes
    def build_qa(self,chunks:list[Chunk])->list[Probe]:
        probes=[]
        for c in chunks:
            sents=_sent_split(c.text)
            if len(sents)<2:continue
            q=f"Based on: \"{sents[0]}\" — what follows?"
            gold=sents[1]
            qid=hashlib.sha256(q.encode()).hexdigest()[:12]
            probes.append(Probe(
                qid=qid,prompt=q,gold=gold,eid=c.eid,kind="qa",
            ))
        return probes
    def build_all(self,chunks:list[Chunk])->list[Probe]:
        return self.build_cloze(chunks)+self.build_qa(chunks)
