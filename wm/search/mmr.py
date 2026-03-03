from __future__ import annotations
import math,re
from collections import Counter

def _tokenize(txt:str)->list[str]:
    return re.findall(r'\w+',txt.lower())

def _tf_vecs(docs:list[str])->list[dict[str,float]]:
    vecs=[]
    for d in docs:
        ws=_tokenize(d)
        tf=Counter(ws)
        tot=max(len(ws),1)
        vecs.append({w:c/tot for w,c in tf.items()})
    return vecs

def _cosine(a:dict[str,float],b:dict[str,float])->float:
    keys=set(a)|set(b)
    dot=sum(a.get(k,0)*b.get(k,0) for k in keys)
    na=math.sqrt(sum(v*v for v in a.values()))
    nb=math.sqrt(sum(v*v for v in b.values()))
    if na<1e-9 or nb<1e-9:return 0.0
    return dot/(na*nb)

def mmr_select(docs:list[str],query:str,k:int=20,lam:float=0.7)->list[int]:
    if not docs:return []
    k=min(k,len(docs))
    all_docs=[query]+docs
    vecs=_tf_vecs(all_docs)
    qv=vecs[0]
    dvecs=vecs[1:]
    selected:list[int]=[]
    remaining=list(range(len(docs)))
    for _ in range(k):
        best_i,best_s=-1,-float('inf')
        for i in remaining:
            rel=_cosine(dvecs[i],qv)
            red=0.0
            if selected:
                red=max(_cosine(dvecs[i],dvecs[j]) for j in selected)
            s=lam*rel-(1-lam)*red
            if s>best_s:
                best_s=s
                best_i=i
        if best_i<0:break
        selected.append(best_i)
        remaining.remove(best_i)
    return selected
