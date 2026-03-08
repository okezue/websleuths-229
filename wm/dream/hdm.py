from __future__ import annotations
import torch
import torch.nn.functional as F

def hard_dream_mine(model,teacher,bank,tok,pool_n:int=50,
                    topk:int=10,dev:torch.device|None=None)->list[tuple[str,str,float]]:
    if dev is None:
        dev=next(model.parameters()).device
    pairs=bank.sample_pool(pool_n)
    if not pairs:return []
    texts=[p[0] for p in pairs]
    bks=[p[1] for p in pairs]
    temps=[bank._t.get(bk,1.5) for bk in bks]
    enc=tok(texts,return_tensors="pt",truncation=True,
            max_length=bank._ml,padding=True)
    enc={k:v.to(dev) for k,v in enc.items()}
    flt={k:v for k,v in enc.items() if k in ("input_ids","attention_mask")}
    model.eval();teacher.eval()
    with torch.no_grad():
        s_out=model(**flt)
        t_out=teacher(**flt)
    s_log=s_out.logits;t_log=t_out.logits
    scores=[]
    for i in range(len(texts)):
        t=temps[i]
        s=F.log_softmax(s_log[i]/t,dim=-1)
        tt=F.softmax(t_log[i]/t,dim=-1)
        kl=F.kl_div(s,tt,reduction="sum").item()*(t**2)
        scores.append((texts[i],bks[i],kl))
    scores.sort(key=lambda x:x[2],reverse=True)
    return scores[:topk]
