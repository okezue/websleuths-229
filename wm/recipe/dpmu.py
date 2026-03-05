from __future__ import annotations
import torch
from torch.optim import AdamW
from wm.types import TrainResult
from wm.recipe.base import weighted_ce,dream_kl,grad_vec,set_grad,WeightedLMCollator,make_ep_dl
from wm.dream.buffer import DreamBuffer

def _project(g_ep:torch.Tensor,g_dr:torch.Tensor)->torch.Tensor:
    dot=(g_dr*g_ep).sum()
    if dot>=0:
        return g_ep
    return g_ep-(dot/(g_dr.norm()**2+1e-12))*g_dr

def _project_multi(g_ep:torch.Tensor,G:list[torch.Tensor],
                   iters:int=50,lr_mu:float=0.5)->torch.Tensor:
    if not G:return g_ep
    A=torch.stack(G)
    dots=A@g_ep
    if (dots>=0).all():return g_ep
    mu=torch.zeros(len(G),device=g_ep.device)
    for _ in range(iters):
        g=g_ep+A.T@mu
        v=A@g
        for j in range(len(G)):
            if v[j]<0:
                mu[j]=mu[j]-v[j]/(A[j]@A[j]+1e-12)
            mu[j]=max(0.0,mu[j].item())
        mu=torch.tensor([mu[j].item() if isinstance(mu[j],torch.Tensor) else mu[j] for j in range(len(G))],device=g_ep.device)
    return g_ep+A.T@mu

class DPMURunner:
    def __init__(self,lr=2e-4,max_steps=100,bs=4,temp=2.0,
                 n_dream_grads=1,max_len=512,dream_n=4,dream_len=128):
        self.lr=lr;self.ms=max_steps;self.bs=bs;self.temp=temp
        self.m=n_dream_grads;self.max_len=max_len
        self.dn=dream_n;self.dl=dream_len
    def run(self,model,teacher,ds,dream_prompts:list[str],tok)->TrainResult:
        dev=next(model.parameters()).device
        teacher=teacher.to(dev);teacher.eval();model.train()
        opt=AdamW([p for p in model.parameters() if p.requires_grad],lr=self.lr)
        col=WeightedLMCollator(tok)
        loader=make_ep_dl(ds,tok,col,self.bs,self.max_len)
        dbuf=DreamBuffer(dream_prompts,tok,self.dl,self.dn)
        tot_loss,tot_dl,steps=0.0,0.0,0
        hist=[]
        for batch in loader:
            if self.ms>0 and steps>=self.ms:break
            batch={k:v.to(dev) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
            ids=batch["input_ids"];mask=batch["attention_mask"];w=batch["weights"]
            out=model(input_ids=ids,attention_mask=mask)
            l_ep=weighted_ce(out.logits,ids,mask,w)
            opt.zero_grad()
            l_ep.backward()
            g_ep=grad_vec(model).clone()
            G_dr=[];_sdl=0.0
            for _ in range(self.m):
                opt.zero_grad()
                d_inp=dbuf.sample(dev)
                if d_inp is None:break
                flt={k:v for k,v in d_inp.items() if k in ("input_ids","attention_mask")}
                s_out=model(**flt)
                with torch.no_grad():
                    t_out=teacher(**flt)
                l_dr=dream_kl(s_out.logits,t_out.logits,self.temp)
                l_dr.backward()
                G_dr.append(grad_vec(model).clone())
                tot_dl+=l_dr.item();_sdl+=l_dr.item()
            opt.zero_grad()
            if len(G_dr)==1:
                g_star=_project(g_ep,G_dr[0])
            elif len(G_dr)>1:
                g_star=_project_multi(g_ep,G_dr)
            else:
                g_star=g_ep
            i=0
            for p in model.parameters():
                if p.requires_grad:
                    n=p.numel()
                    p.grad=g_star[i:i+n].view_as(p).clone()
                    i+=n
            opt.step()
            tot_loss+=l_ep.item();steps+=1
            hist.append({"loss":l_ep.item(),"dream_loss":_sdl/max(len(G_dr),1),"n_grads":len(G_dr)})
        avg=tot_loss/max(steps,1)
        avg_dl=tot_dl/max(steps*self.m,1)
        return TrainResult(loss=avg,steps=steps,lr=self.lr,
                           dream_loss=avg_dl,extras={"n_dream_grads":self.m},
                           history=hist)
