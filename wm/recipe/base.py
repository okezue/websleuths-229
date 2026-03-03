from __future__ import annotations
import math,torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

def weighted_ce(logits:torch.Tensor,ids:torch.Tensor,
                mask:torch.Tensor,w:torch.Tensor)->torch.Tensor:
    B,S,V=logits.shape
    sl=logits[...,:-1,:].contiguous().view(-1,V)
    tgt=ids[...,1:].contiguous().view(-1)
    loss=CrossEntropyLoss(reduction="none")(sl,tgt).view(B,S-1)
    m=mask[...,1:].float()
    per=((loss*m).sum(1))/(m.sum(1).clamp(min=1))
    return (w*per).sum()/w.sum().clamp(min=1e-8)

def dream_kl(s_logits:torch.Tensor,t_logits:torch.Tensor,
             temp:float=2.0)->torch.Tensor:
    s=F.log_softmax(s_logits/temp,dim=-1)
    t=F.softmax(t_logits/temp,dim=-1)
    return F.kl_div(s,t,reduction="batchmean")*(temp**2)

def evidence_eps(w:torch.Tensor,n:int,
                 eps_min:float=0.01,alpha:float=0.5)->float:
    return eps_min+alpha*w.mean().item()*math.log(1+n)

def grad_vec(model)->torch.Tensor:
    gs=[]
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            gs.append(p.grad.detach().view(-1))
    return torch.cat(gs) if gs else torch.zeros(1)

def set_grad(model,vec:torch.Tensor):
    i=0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            n=p.grad.numel()
            p.grad.copy_(vec[i:i+n].view_as(p.grad))
            i+=n

def lora_param_count(model)->int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class WeightedLMCollator:
    def __init__(self,tok,mlm=False):
        self._inner=DataCollatorForLanguageModeling(tok,mlm=mlm)
    def __call__(self,features):
        ws=[f.pop("_wt",1.0) for f in features]
        batch=self._inner(features)
        batch["weights"]=torch.tensor(ws,dtype=torch.float32)
        return batch

def make_ep_dl(ds,tok,collator,bs=4,max_len=512):
    def tkfn(ex):
        t=ex.get("text","")
        if not t:
            t=ex.get("prompt","")+"\n"+ex.get("completion","")
        enc=tok(t,truncation=True,max_length=max_len,padding=False)
        enc["_wt"]=ex.get("authority",1.0)
        return enc
    td=ds.map(tkfn,remove_columns=ds.column_names)
    td.set_format("torch",columns=["input_ids","attention_mask","_wt"])
    return DataLoader(td,batch_size=bs,shuffle=True,collate_fn=collator,drop_last=False)
