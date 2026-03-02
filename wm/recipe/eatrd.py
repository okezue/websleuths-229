from __future__ import annotations
import torch
from torch.optim import AdamW
from wm.types import TrainResult
from wm.recipe.base import weighted_ce,dream_kl,evidence_eps,WeightedLMCollator,make_ep_dl
from wm.dream.buffer import DreamBuffer

class EATRDRunner:
    def __init__(self,lr=2e-4,max_steps=100,bs=4,temp=2.0,
                 eps_min=0.01,alpha=0.5,rho=0.01,lam_init=1.0,
                 max_len=512,dream_n=4,dream_len=128):
        self.lr=lr;self.ms=max_steps;self.bs=bs;self.temp=temp
        self.eps_min=eps_min;self.alpha=alpha;self.rho=rho
        self.lam=lam_init;self.max_len=max_len
        self.dn=dream_n;self.dl=dream_len
    def run(self,model,teacher,ds,dream_prompts:list[str],tok)->TrainResult:
        dev=next(model.parameters()).device
        teacher=teacher.to(dev);teacher.eval();model.train()
        opt=AdamW([p for p in model.parameters() if p.requires_grad],lr=self.lr)
        col=WeightedLMCollator(tok)
        dl=make_ep_dl(ds,tok,col,self.bs,self.max_len)
        dbuf=DreamBuffer(dream_prompts,tok,self.dl,self.dn)
        n_ep=len(ds)
        all_w=torch.tensor([ds[i].get("authority",1.0) for i in range(len(ds))])
        eps_k=evidence_eps(all_w,n_ep,self.eps_min,self.alpha)
        lam=self.lam
        tot_loss,tot_dl,steps=0.0,0.0,0
        for batch in dl:
            if self.ms>0 and steps>=self.ms:break
            batch={k:v.to(dev) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
            ids=batch["input_ids"];mask=batch["attention_mask"];w=batch["weights"]
            out=model(input_ids=ids,attention_mask=mask)
            l_ep=weighted_ce(out.logits,ids,mask,w)
            d_inp=dbuf.sample(dev)
            if d_inp is not None:
                s_out=model(**{k:v for k,v in d_inp.items() if k in ("input_ids","attention_mask")})
                with torch.no_grad():
                    t_out=teacher(**{k:v for k,v in d_inp.items() if k in ("input_ids","attention_mask")})
                l_dr=dream_kl(s_out.logits,t_out.logits,self.temp)
            else:
                l_dr=torch.tensor(0.0,device=dev)
            loss=l_ep+lam*l_dr
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                lam=max(0.0,lam+self.rho*(l_dr.item()-eps_k))
            tot_loss+=l_ep.item();tot_dl+=l_dr.item();steps+=1
        avg_loss=tot_loss/max(steps,1)
        avg_dl=tot_dl/max(steps,1)
        return TrainResult(loss=avg_loss,steps=steps,lr=self.lr,
                           dream_loss=avg_dl,extras={"lambda":lam,"eps_k":eps_k})
