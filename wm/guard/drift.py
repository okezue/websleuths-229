from __future__ import annotations
import torch,torch.nn.functional as F

def drift_kl(model_new,model_old,tok,anchors:list[str],
             max_len:int=128)->float:
    model_new.eval()
    model_old.eval()
    dev=next(model_new.parameters()).device
    total,n=0.0,0
    with torch.no_grad():
        for a in anchors:
            enc={k:v.to(dev) for k,v in tok(a,return_tensors="pt",truncation=True,max_length=max_len).items()}
            lo_new=model_new(**enc).logits
            lo_old=model_old(**enc).logits
            p=F.softmax(lo_old,dim=-1)
            q=F.log_softmax(lo_new,dim=-1)
            kl=F.kl_div(q,p,reduction="batchmean").clamp_min(0.0).item()
            total+=kl
            n+=1
    return total/max(n,1)
