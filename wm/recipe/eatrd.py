from __future__ import annotations
import torch
from torch.optim import AdamW
from wm.types import TrainResult
from wm.recipe.base import weighted_ce,dream_kl,multi_temp_dream_kl,evidence_eps,WeightedLMCollator,make_ep_dl
from wm.dream.buffer import DreamBuffer

class EATRDRunner:
    def __init__(self,lr=2e-4,max_steps=100,bs=4,temp=2.0,
                 eps_min=0.01,alpha=0.5,rho=0.01,lam_init=1.0,
                 max_len=512,dream_n=4,dream_len=128,
                 d_targ=0.5,lam_floor=0.01,lam_ceil=10.0,use_pi=True,
                 pi_warmup_frac=0.15):
        self.lr=lr;self.ms=max_steps;self.bs=bs;self.temp=temp
        self.eps_min=eps_min;self.alpha=alpha;self.rho=rho
        self.lam=lam_init;self.max_len=max_len
        self.dn=dream_n;self.dl=dream_len
        self.d_targ=d_targ;self.lam_floor=lam_floor
        self.lam_ceil=lam_ceil;self.use_pi=use_pi
        self.pi_warmup_frac=pi_warmup_frac
    def run(self,model,teacher,ds,dream_prompts:list[str],tok,
            dbank=None)->TrainResult:
        dev=next(model.parameters()).device
        if teacher is not None:
            teacher=teacher.to(dev)
            teacher.eval()
        model.train()
        opt=AdamW([p for p in model.parameters() if p.requires_grad],lr=self.lr)
        col=WeightedLMCollator(tok)
        dl=make_ep_dl(ds,tok,col,self.bs,self.max_len)
        dbuf=DreamBuffer(dream_prompts,tok,self.dl,self.dn) if not dbank else None
        n_ep=len(ds)
        all_w=torch.tensor([ds[i].get("authority",1.0) for i in range(len(ds))])
        eps_k=evidence_eps(all_w,n_ep,self.eps_min,self.alpha)
        lam=self.lam
        warmup_steps=max(1,int(self.ms*self.pi_warmup_frac)) if self.ms>0 else 10
        tot_loss,tot_dl,steps=0.0,0.0,0
        hist=[]
        done=False
        while not done:
            for batch in dl:
                if self.ms>0 and steps>=self.ms:
                    done=True;break
                batch={k:v.to(dev) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
                ids=batch["input_ids"];mask=batch["attention_mask"];w=batch["weights"]
                out=model(input_ids=ids,attention_mask=mask)
                l_ep=weighted_ce(out.logits,ids,mask,w)
                if teacher is None:
                    l_dr=torch.tensor(0.0,device=dev)
                elif dbank is not None:
                    st=dbank.sample_with_temps(dev)
                    if st is not None:
                        d_inp,temps=st
                        flt={k:v for k,v in d_inp.items() if k in ("input_ids","attention_mask")}
                        s_out=model(**flt)
                        with torch.no_grad():
                            t_out=teacher(**flt)
                        l_dr=multi_temp_dream_kl(s_out.logits,t_out.logits,temps)
                    else:
                        l_dr=torch.tensor(0.0,device=dev)
                else:
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
                    d=l_dr.item()
                    if self.use_pi and steps>=warmup_steps:
                        if d<self.d_targ/1.5:
                            lam=lam*0.9
                        elif d>1.5*self.d_targ:
                            lam=lam*1.5
                        lam=max(self.lam_floor,min(self.lam_ceil,lam))
                    elif not self.use_pi:
                        lam=max(0.0,lam+self.rho*(d-eps_k))
                tot_loss+=l_ep.item();tot_dl+=d;steps+=1
                hist.append({"loss":l_ep.item(),"dream_loss":d,"lambda":lam,
                             "eps_k":eps_k,"d_targ":self.d_targ,"warmup":steps<warmup_steps})
            if self.ms<=0:break
        avg_loss=tot_loss/max(steps,1)
        avg_dl=tot_dl/max(steps,1)
        return TrainResult(loss=avg_loss,steps=steps,lr=self.lr,
                           dream_loss=avg_dl,extras={"lambda":lam,"eps_k":eps_k,
                                                      "d_targ":self.d_targ,"use_pi":self.use_pi},
                           history=hist)
