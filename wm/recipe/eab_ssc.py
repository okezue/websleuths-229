from __future__ import annotations
import copy,torch
from torch.optim import AdamW
from collections import OrderedDict
from wm.types import TrainResult
from wm.recipe.base import weighted_ce,dream_kl,multi_temp_dream_kl,WeightedLMCollator,make_ep_dl
from wm.dream.buffer import DreamBuffer

class AdapterBank:
    def __init__(self):
        self._adapters:list[tuple[float,OrderedDict]]=[]
    def add(self,model,weight:float=1.0):
        sd=OrderedDict()
        for n,p in model.named_parameters():
            if p.requires_grad:
                sd[n]=p.detach().cpu().clone()
        self._adapters.append((weight,sd))
    @property
    def size(self)->int:
        return len(self._adapters)
    def get_all(self)->list[tuple[float,OrderedDict]]:
        return self._adapters
    def clear(self):
        self._adapters.clear()

def _find_lora_pairs(sd:OrderedDict)->dict[str,tuple[str,str]]:
    pairs={}
    for k in sd:
        if "lora_A" in k:
            bk=k.replace("lora_A","lora_B")
            if bk in sd:
                base=k.split(".lora_A")[0]
                pairs[base]=(k,bk)
    return pairs

def _merge_deltas(adapters:list[tuple[float,OrderedDict]],
                  pairs:dict[str,tuple[str,str]])->dict[str,torch.Tensor]:
    tw=sum(w for w,_ in adapters)
    merged={}
    for base,(ak,bk) in pairs.items():
        dws=[]
        for w,sd in adapters:
            A=sd[ak];B=sd[bk]
            dws.append((w/tw)*(B@A))
        merged[base]=sum(dws)
    return merged

def _svd_trunc(dw:torch.Tensor,r:int)->tuple[torch.Tensor,torch.Tensor]:
    U,S,Vh=torch.linalg.svd(dw.float(),full_matrices=False)
    r=min(r,len(S))
    sqS=S[:r].sqrt()
    A=torch.diag(sqS)@Vh[:r,:]
    B=U[:,:r]@torch.diag(sqS)
    return A.to(dw.dtype),B.to(dw.dtype)

def _spectral_reg(model,pairs:dict[str,tuple[str,str]],
                  tau:float=1.0)->torch.Tensor:
    loss=torch.tensor(0.0)
    for base,(ak,bk) in pairs.items():
        A=B=None
        for n,p in model.named_parameters():
            if n==ak:A=p
            if n==bk:B=p
        if A is not None and B is not None:
            dw=B@A
            s=torch.linalg.svdvals(dw.float())
            loss=loss+(s[0]-tau)**2
    return loss

class SleepConsolidator:
    def __init__(self,rank=16,beta=0.1,gamma=0.01,tau=1.0,
                 refine_steps=20,lr=1e-4,temp=2.0,dream_n=4,dream_len=128):
        self.r=rank;self.beta=beta;self.gamma=gamma;self.tau=tau
        self.rs=refine_steps;self.lr=lr;self.temp=temp
        self.dn=dream_n;self.dl=dream_len
    def consolidate(self,bank:AdapterBank,base_model,
                    dream_prompts:list[str],tok)->None:
        adapters=bank.get_all()
        if not adapters:return
        ref_sd=adapters[0][1]
        pairs=_find_lora_pairs(ref_sd)
        if not pairs:return
        merged=_merge_deltas(adapters,pairs)
        dev=next(base_model.parameters()).device
        for base,(ak,bk) in pairs.items():
            dw=merged[base]
            A,B=_svd_trunc(dw,self.r)
            for n,p in base_model.named_parameters():
                if n==ak:p.data.copy_(A.to(dev))
                if n==bk:p.data.copy_(B.to(dev))
        if not dream_prompts or self.rs<=0:return
        dbuf=DreamBuffer(dream_prompts,tok,self.dl,self.dn)
        merge_model=copy.deepcopy(base_model).to(dev)
        for base,(ak,bk) in pairs.items():
            dw=merged[base].to(dev)
            Af,Bf=_svd_trunc(dw,self.r)
            for n,p in merge_model.named_parameters():
                if n==ak:p.data.copy_(Af)
                if n==bk:p.data.copy_(Bf)
        merge_model.eval()
        opt=AdamW([p for p in base_model.parameters() if p.requires_grad],lr=self.lr)
        base_model.train()
        for _ in range(self.rs):
            d_inp=dbuf.sample(dev)
            if d_inp is None:break
            flt={k:v for k,v in d_inp.items() if k in ("input_ids","attention_mask")}
            s_out=base_model(**flt)
            with torch.no_grad():
                m_out=merge_model(**flt)
            l_dist=dream_kl(s_out.logits,m_out.logits,self.temp)
            l_spec=_spectral_reg(base_model,pairs,self.tau)
            loss=l_dist+self.gamma*l_spec
            opt.zero_grad()
            loss.backward()
            opt.step()
        del merge_model

class EABSSCRunner:
    def __init__(self,lr=2e-4,max_steps=100,bs=4,temp=2.0,
                 dream_weight=0.5,max_len=512,dream_n=4,dream_len=128):
        self.lr=lr;self.ms=max_steps;self.bs=bs;self.temp=temp
        self.dw=dream_weight;self.max_len=max_len
        self.dn=dream_n;self.dl=dream_len
    def day(self,model,teacher,ds,dream_prompts:list[str],tok,
            dbank=None)->TrainResult:
        dev=next(model.parameters()).device
        teacher=teacher.to(dev);teacher.eval();model.train()
        opt=AdamW([p for p in model.parameters() if p.requires_grad],lr=self.lr)
        col=WeightedLMCollator(tok)
        loader=make_ep_dl(ds,tok,col,self.bs,self.max_len)
        dbuf=DreamBuffer(dream_prompts,tok,self.dl,self.dn) if not dbank else None
        tot_loss,tot_dl,steps=0.0,0.0,0
        hist=[]
        done=False
        while not done:
            for batch in loader:
                if self.ms>0 and steps>=self.ms:
                    done=True;break
                batch={k:v.to(dev) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
                ids=batch["input_ids"];mask=batch["attention_mask"];w=batch["weights"]
                out=model(input_ids=ids,attention_mask=mask)
                l_ep=weighted_ce(out.logits,ids,mask,w)
                if dbank is not None:
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
                        flt={k:v for k,v in d_inp.items() if k in ("input_ids","attention_mask")}
                        s_out=model(**flt)
                        with torch.no_grad():
                            t_out=teacher(**flt)
                        l_dr=dream_kl(s_out.logits,t_out.logits,self.temp)
                    else:
                        l_dr=torch.tensor(0.0,device=dev)
                loss=l_ep+self.dw*l_dr
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot_loss+=l_ep.item();tot_dl+=l_dr.item();steps+=1
                hist.append({"loss":l_ep.item(),"dream_loss":l_dr.item()})
            if self.ms<=0:break
        return TrainResult(loss=tot_loss/max(steps,1),steps=steps,
                           lr=self.lr,dream_loss=tot_dl/max(steps,1),
                           history=hist)
