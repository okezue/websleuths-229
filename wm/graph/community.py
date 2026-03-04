from __future__ import annotations
import math,re
from hashlib import sha256
from collections import Counter
from wm.types import Claim,Community

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

def _mean_vec(vecs:list[dict[str,float]])->dict[str,float]:
    if not vecs:return {}
    keys=set()
    for v in vecs:keys|=set(v)
    n=len(vecs)
    return {k:sum(v.get(k,0) for v in vecs)/n for k in keys}

def _top_entity(claims:list[Claim])->str:
    cnt=Counter()
    for c in claims:
        for e in c.entities:cnt[e]+=1
    return cnt.most_common(1)[0][0] if cnt else "unknown"

def _agglom(dists:list[list[float]],thresh:float=0.5)->list[list[int]]:
    n=len(dists)
    labels=list(range(n))
    merged=True
    while merged:
        merged=False
        clusters:dict[int,list[int]]={}
        for i,l in enumerate(labels):
            clusters.setdefault(l,[]).append(i)
        cids=sorted(clusters.keys())
        best_d,best_pair=float('inf'),None
        for ci in range(len(cids)):
            for cj in range(ci+1,len(cids)):
                a,b=clusters[cids[ci]],clusters[cids[cj]]
                d=sum(dists[i][j] for i in a for j in b)/(len(a)*len(b))
                if d<best_d:
                    best_d=d
                    best_pair=(cids[ci],cids[cj])
        if best_pair and best_d<thresh:
            lo,hi=best_pair
            for i in range(n):
                if labels[i]==hi:labels[i]=lo
            merged=True
    clusters:dict[int,list[int]]={}
    for i,l in enumerate(labels):
        clusters.setdefault(l,[]).append(i)
    return list(clusters.values())

def detect_communities(claims:list[Claim],thresh:float=0.5,max_claims:int=200)->list[Community]:
    if not claims:return []
    if len(claims)>max_claims:
        import random
        rng=random.Random(42)
        claims=rng.sample(claims,max_claims)
    if len(claims)==1:
        cid=sha256(claims[0].text.encode()).hexdigest()[:12]
        tv=_tf_vecs([claims[0].text])
        return [Community(coid=cid,label=_top_entity(claims),
                          members=[claims[0].cid],centroid=tv[0])]
    texts=[c.text for c in claims]
    vecs=_tf_vecs(texts)
    n=len(claims)
    dists=[[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1,n):
            d=1.0-_cosine(vecs[i],vecs[j])
            dists[i][j]=d
            dists[j][i]=d
    try:
        from scipy.cluster.hierarchy import fcluster,linkage
        import numpy as np
        cond=[]
        for i in range(n):
            for j in range(i+1,n):
                cond.append(max(0.0,dists[i][j]))
        cond=np.array(cond,dtype=np.float64)
        cond=np.clip(cond,0.0,None)
        Z=linkage(cond,method='average')
        labs=fcluster(Z,t=thresh,criterion='distance')
        clusters:dict[int,list[int]]={}
        for i,l in enumerate(labs):
            clusters.setdefault(l,[]).append(i)
        groups=list(clusters.values())
    except ImportError:
        groups=_agglom(dists,thresh)
    result=[]
    for grp in groups:
        grp_claims=[claims[i] for i in grp]
        grp_vecs=[vecs[i] for i in grp]
        centroid=_mean_vec(grp_vecs)
        label=_top_entity(grp_claims)
        members=[c.cid for c in grp_claims]
        coid=sha256(",".join(sorted(members)).encode()).hexdigest()[:12]
        result.append(Community(coid=coid,label=label,members=members,
                                centroid=centroid))
    return result
