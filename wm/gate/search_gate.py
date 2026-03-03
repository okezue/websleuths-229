from __future__ import annotations
import re,math
from collections import Counter
from wm.types import SearchGateResult,Community
from wm.cfg import SearchGateCfg

def _tokenize(txt:str)->list[str]:
    return re.findall(r'\w+',txt.lower())

def _tf_vec(txt:str)->dict[str,float]:
    ws=_tokenize(txt)
    tf=Counter(ws)
    tot=max(len(ws),1)
    return {w:c/tot for w,c in tf.items()}

def _cosine(a:dict[str,float],b:dict[str,float])->float:
    keys=set(a)|set(b)
    dot=sum(a.get(k,0)*b.get(k,0) for k in keys)
    na=math.sqrt(sum(v*v for v in a.values()))
    nb=math.sqrt(sum(v*v for v in b.values()))
    if na<1e-9 or nb<1e-9:return 0.0
    return dot/(na*nb)

_FRESH_RE=re.compile(r'2024|2025|2026|latest|recent|current|breaking',re.I)

def _freshness(query:str)->float:
    return 1.0 if _FRESH_RE.search(query) else 0.0

def _param_match(query:str,communities:list[Community])->float:
    if not communities:return 0.0
    qv=_tf_vec(query)
    return max(_cosine(qv,co.centroid) for co in communities)

def _uncertainty_from_logits(logits)->float:
    import torch
    lp=torch.log_softmax(logits[:,-1,:],dim=-1)
    p=lp.exp()
    ent=-(p*lp).sum(dim=-1).mean().item()
    vocab=logits.shape[-1]
    return ent/math.log(max(vocab,2))

class SearchGate:
    def __init__(self,cfg:SearchGateCfg):
        self._cfg=cfg
    def check(self,query:str,communities:list[Community]|None=None,
              model=None,tok=None,tau:float|None=None)->SearchGateResult:
        if not self._cfg.enabled:
            return SearchGateResult(should_search=True,reason="gate disabled")
        unc=1.0
        if model is not None and tok is not None:
            import torch
            model.eval()
            dev=next(model.parameters()).device
            enc={k:v.to(dev) for k,v in tok(query,return_tensors="pt",
                 truncation=True,max_length=128).items()}
            with torch.no_grad():
                out=model(**enc)
            unc=_uncertainty_from_logits(out.logits)
        fr=_freshness(query)
        pm=_param_match(query,communities or [])
        sc=self._cfg.a*unc+self._cfg.b*fr-self._cfg.c*pm
        th=tau if tau is not None else self._cfg.tau_search
        should=sc>=th
        return SearchGateResult(
            should_search=should,score=sc,uncertainty=unc,
            freshness=fr,param_match=pm,
            reason=f"score={sc:.3f} vs tau={th:.3f}")
