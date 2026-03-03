from __future__ import annotations
import re
from hashlib import sha256
from wm.types import Claim,Entity

def _sent_split(txt:str)->list[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+',txt) if s.strip()]

_VERB_RE=re.compile(r'\b(is|are|was|were|has|have|had|does|do|did|can|could|will|would|shall|should|may|might|must|contains|involves|produces|causes|results|shows|indicates|demonstrates|provides|requires|includes|represents|means|leads|affects|creates|forms|generates|consists|occurs|exists|remains|becomes|appears|seems|makes|gives|takes|uses|finds|goes|comes|says|gets|knows|thinks|sees|looks|works|runs|holds|keeps|plays|moves|lives|believes|brings|happens|writes|stands|loses|pays|meets|drives|sets|grows|opens|walks|wins|offers|remembers|loves|considers|buys|waits|serves|dies|sends|expects|builds|stays|falls|cuts|reaches|kills|raises|passes|sells|decides|returns|explains|hopes|develops|carries|breaks|receives|agrees|supports|hits|eats|covers|catches|draws|chooses|confirmed|determined|collected|recovered|places|conducted|reported|identified|revealed|detected|analyzed|established|measured|observed|tested|examined|belong|belongs)\b',re.I)
_ENT_RE=re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|[A-Z]{2,}|\d[\d,.]+)\b')

def _is_claim(s:str)->bool:
    if len(s)<30 or len(s)>300:return False
    if s.endswith('?'):return False
    if not _VERB_RE.search(s):return False
    return True

def extract_claims(txt:str,eid:str="",chunk_idx:int=0)->list[Claim]:
    sents=_sent_split(txt)
    claims=[]
    for s in sents:
        if not _is_claim(s):continue
        norm=s.strip().lower()
        cid=sha256(norm.encode()).hexdigest()[:12]
        ents=[m.group() for m in _ENT_RE.finditer(s)]
        claims.append(Claim(cid=cid,text=s,eid=eid,chunk_idx=chunk_idx,
                            entities=ents))
    return claims

def extract_entities(claims:list[Claim])->list[Entity]:
    seen={}
    for c in claims:
        for e in c.entities:
            if e not in seen:
                nid=sha256(e.encode()).hexdigest()[:12]
                seen[e]=Entity(nid=nid,name=e)
    return list(seen.values())
