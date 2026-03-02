from __future__ import annotations
import math,re
from collections import Counter
from urllib.parse import urlparse
from wm.types import Episode,GateResult
from wm.cfg import GateCfg
import torch

def _domain(url:str)->str:
    return urlparse(url).netloc.lower().lstrip("www.")

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

def _pairwise_consistency(eps:list[Episode])->float:
    if len(eps)<2:return 1.0
    vecs=_tf_vecs([e.body for e in eps])
    sims=[]
    for i in range(len(vecs)):
        for j in range(i+1,len(vecs)):
            sims.append(_cosine(vecs[i],vecs[j]))
    return sum(sims)/max(len(sims),1)

def _model_uncertainty(model,tok,eps:list[Episode],max_len:int=128)->float:
    if model is None or tok is None:return 1.0
    model.eval()
    ents=[]
    dev=next(model.parameters()).device
    with torch.no_grad():
        for e in eps[:3]:
            txt=e.body[:500]
            enc={k:v.to(dev) for k,v in tok(txt,return_tensors="pt",truncation=True,max_length=max_len).items()}
            out=model(**enc)
            lp=torch.log_softmax(out.logits[:,-1,:],dim=-1)
            p=lp.exp()
            ent=-(p*lp).sum(dim=-1).mean().item()
            ents.append(ent)
    return sum(ents)/max(len(ents),1)

class EpisodeGate:
    def __init__(self,cfg:GateCfg):
        self._cfg=cfg
    def check(self,eps:list[Episode],model=None,tok=None)->GateResult:
        if not self._cfg.enabled:
            return GateResult(accept=True,reason="gating disabled")
        doms=set(_domain(e.url) for e in eps)
        ns=len(doms)
        if ns<self._cfg.min_sources:
            return GateResult(accept=False,n_sources=ns,
                reason=f"only {ns} sources, need {self._cfg.min_sources}")
        cons=_pairwise_consistency(eps)
        if cons<self._cfg.min_consistency:
            return GateResult(accept=False,n_sources=ns,consistency=cons,
                reason=f"consistency {cons:.3f} < {self._cfg.min_consistency}")
        unc=_model_uncertainty(model,tok,eps)
        if unc<self._cfg.uncertainty_thresh and model is not None:
            return GateResult(accept=False,n_sources=ns,consistency=cons,
                uncertainty=unc,reason=f"uncertainty {unc:.3f} < {self._cfg.uncertainty_thresh}")
        return GateResult(accept=True,n_sources=ns,consistency=cons,uncertainty=unc)
