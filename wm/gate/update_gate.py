from __future__ import annotations
import math,re
from collections import Counter
from urllib.parse import urlparse
from wm.types import SearchResult,UpdateGateResult
from wm.cfg import UpdateGateCfg

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

def _pairwise_consistency(texts:list[str])->float:
    if len(texts)<2:return 1.0
    vecs=_tf_vecs(texts)
    sims=[]
    for i in range(len(vecs)):
        for j in range(i+1,len(vecs)):
            sims.append(_cosine(vecs[i],vecs[j]))
    return sum(sims)/max(len(sims),1)

def _sigmoid(x:float)->float:
    return 1.0/(1.0+math.exp(-x))

def _readiness(sr:SearchResult,cfg:UpdateGateCfg)->float:
    doms=set(_domain(u) for u in sr.sources if u)
    ns=len(doms)
    texts=[c.text for c in sr.claims]
    div=1.0-_pairwise_consistency(texts) if len(texts)>=2 else 0.0
    agr=_pairwise_consistency(texts) if len(texts)>=2 else 0.0
    rel=sum(c.confidence for c in sr.claims)/max(len(sr.claims),1)
    x=cfg.a1*math.log(1+ns)+cfg.a2*div+cfg.a3*agr+cfg.a4*rel
    return _sigmoid(x)

def _novelty_nll(sr:SearchResult,model,tok)->float:
    if model is None or tok is None:return 1.0
    import torch
    model.eval()
    dev=next(model.parameters()).device
    nlls=[]
    with torch.no_grad():
        for c in sr.claims[:20]:
            enc={k:v.to(dev) for k,v in tok(c.text,return_tensors="pt",
                 truncation=True,max_length=256).items()}
            out=model(**enc,labels=enc["input_ids"])
            nlls.append(out.loss.item())
    return sum(nlls)/max(len(nlls),1) if nlls else 1.0

class UpdateGate:
    def __init__(self,cfg:UpdateGateCfg):
        self._cfg=cfg
    def check(self,sr:SearchResult,model=None,tok=None,
              tau_ready:float|None=None,tau_novel:float|None=None)->UpdateGateResult:
        if not self._cfg.enabled:
            return UpdateGateResult(should_update=True,reason="gate disabled")
        rdy=_readiness(sr,self._cfg)
        nov=_novelty_nll(sr,model,tok)
        tr=tau_ready if tau_ready is not None else self._cfg.tau_ready
        tn=tau_novel if tau_novel is not None else self._cfg.tau_novel
        should=rdy>=tr and nov>=tn
        return UpdateGateResult(
            should_update=should,readiness=rdy,novelty=nov,
            reason=f"ready={rdy:.3f}>={tr:.3f} novel={nov:.3f}>={tn:.3f}")
