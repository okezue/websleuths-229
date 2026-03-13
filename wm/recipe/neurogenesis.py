from __future__ import annotations
import copy,logging,math,torch
import torch.nn.functional as F
from torch.optim import AdamW
from collections import OrderedDict
from wm.types import TrainResult
from wm.recipe.base import weighted_ce,dream_kl,multi_temp_dream_kl,evidence_eps,WeightedLMCollator,make_ep_dl,grad_vec
from wm.dream.buffer import DreamBuffer

log=logging.getLogger(__name__)

class AdapterSlot:
    def __init__(self,idx:int,sd:OrderedDict,domain:str="",topic:str="",
                 frozen:bool=False):
        self.idx=idx
        self.sd=sd
        self.domain=domain
        self.topic=topic
        self.frozen=frozen
        self.guard_failures=0
        self.ep_loss_residual=0.0

class NeurogenesisBank:
    def __init__(self,max_adapters:int=20):
        self._slots:list[AdapterSlot]=[]
        self._max=max_adapters
        self._next_idx=0
    @property
    def n_adapters(self)->int:
        return len(self._slots)
    @property
    def slots(self)->list[AdapterSlot]:
        return self._slots
    def snapshot_adapter(self,model,domain:str="",topic:str="")->AdapterSlot:
        sd=OrderedDict()
        for n,p in model.named_parameters():
            if p.requires_grad:
                sd[n]=p.detach().cpu().clone()
        slot=AdapterSlot(self._next_idx,sd,domain,topic,frozen=False)
        self._next_idx+=1
        self._slots.append(slot)
        return slot
    def freeze_adapter(self,idx:int):
        for s in self._slots:
            if s.idx==idx:
                s.frozen=True
                break
    def freeze_all(self):
        for s in self._slots:
            s.frozen=True
    def restore_adapter(self,model,idx:int):
        dev=next(model.parameters()).device
        for s in self._slots:
            if s.idx==idx:
                for n,p in model.named_parameters():
                    if n in s.sd:
                        p.data.copy_(s.sd[n].to(dev))
                break
    def should_spawn(self,ep_loss:float,dream_loss:float,
                     guard_failures:int,
                     loss_thresh:float=2.0,dream_thresh:float=1.5,
                     fail_thresh:int=2)->tuple[bool,str]:
        if guard_failures>=fail_thresh:
            return True,"guard_failures"
        if ep_loss>loss_thresh:
            return True,"high_residual_loss"
        if dream_loss>dream_thresh and ep_loss>loss_thresh*0.5:
            return True,"conflict"
        return False,""

def _detect_grad_conflict(model,ep_loss,dream_loss)->float:
    ep_loss.backward(retain_graph=True)
    g_ep=grad_vec(model).clone()
    model.zero_grad()
    dream_loss.backward(retain_graph=True)
    g_dr=grad_vec(model).clone()
    model.zero_grad()
    cos=F.cosine_similarity(g_ep.unsqueeze(0),g_dr.unsqueeze(0)).item()
    return cos

class RankGrowthLoRA:
    def __init__(self,r_step:int=8,max_total_r:int=128,ortho_weight:float=0.01):
        self._rs=r_step
        self._max_r=max_total_r
        self._ortho=ortho_weight
        self._chunks:list[dict[str,tuple[torch.Tensor,torch.Tensor]]]=[]
    @property
    def n_chunks(self)->int:
        return len(self._chunks)
    @property
    def total_rank(self)->int:
        return self.n_chunks*self._rs
    def freeze_current(self,model):
        chunk={}
        for n,p in model.named_parameters():
            if p.requires_grad and "lora_A" in n:
                bk=n.replace("lora_A","lora_B")
                for n2,p2 in model.named_parameters():
                    if n2==bk:
                        chunk[n]=(p.detach().cpu().clone(),p2.detach().cpu().clone())
                        break
        if chunk:
            self._chunks.append(chunk)
        for n,p in model.named_parameters():
            if p.requires_grad:
                p.requires_grad_(False)
    def add_rank_chunk(self,model,peft_model=None):
        if self.total_rank>=self._max_r:
            log.warning("max rank %d reached, not adding",self._max_r)
            return False
        if peft_model is not None:
            try:
                from peft import LoraConfig
                cfg=peft_model.peft_config.get("default")
                if cfg:
                    new_r=cfg.r+self._rs
                    cfg.r=new_r
                    log.info("rank growth: new rank=%d (chunk %d)",new_r,self.n_chunks+1)
            except Exception as e:
                log.warning("peft rank update failed: %s",e)
        for n,p in model.named_parameters():
            if "lora_" in n:
                p.requires_grad_(True)
        return True
    def ortho_penalty(self,model)->torch.Tensor:
        if not self._chunks or self._ortho<=0:
            return torch.tensor(0.0)
        pen=torch.tensor(0.0)
        for n,p in model.named_parameters():
            if p.requires_grad and "lora_A" in n:
                for chunk in self._chunks:
                    if n in chunk:
                        old_a,_=chunk[n]
                        dev=p.device
                        oa=old_a.to(dev)
                        cos=F.cosine_similarity(
                            p.view(1,-1),oa.view(1,-1)).abs()
                        pen=pen+cos
        return self._ortho*pen

class NeurogenesisEATRD:
    def __init__(self,lr=2e-4,max_steps=100,bs=4,temp=2.0,
                 eps_min=0.01,alpha=0.5,rho=0.01,lam_init=1.0,
                 max_len=512,dream_n=4,dream_len=128,
                 d_targ=0.5,lam_floor=0.01,lam_ceil=10.0,use_pi=True,
                 pi_warmup_frac=0.15,
                 mu_init=0.5,mu_floor=0.05,mu_ceil=2.0,
                 spawn_loss_thresh=2.0,spawn_dream_thresh=1.5,
                 spawn_fail_thresh=2,
                 rank_step=8,max_rank=128,ortho_weight=0.01):
        self.lr=lr;self.ms=max_steps;self.bs=bs;self.temp=temp
        self.eps_min=eps_min;self.alpha=alpha;self.rho=rho
        self.lam=lam_init;self.max_len=max_len
        self.dn=dream_n;self.dl=dream_len
        self.d_targ=d_targ;self.lam_floor=lam_floor
        self.lam_ceil=lam_ceil;self.use_pi=use_pi
        self.pi_warmup_frac=pi_warmup_frac
        self.mu=mu_init;self.mu_floor=mu_floor;self.mu_ceil=mu_ceil
        self.slt=spawn_loss_thresh;self.sdt=spawn_dream_thresh
        self.sft=spawn_fail_thresh
        self.rg=RankGrowthLoRA(r_step=rank_step,max_total_r=max_rank,
                                ortho_weight=ortho_weight)
    def run(self,model,teacher,ds,dream_prompts:list[str],tok,
            dbank=None,distill_ds=None,
            bank:NeurogenesisBank|None=None,
            guard_failures:int=0,
            domain:str="",topic:str="")->TrainResult:
        dev=next(model.parameters()).device
        t_dev=next(teacher.parameters()).device
        if torch.cuda.is_available():torch.cuda.empty_cache()
        teacher.eval();model.train()
        if bank is None:
            bank=NeurogenesisBank()
        spawn,reason=bank.should_spawn(
            ep_loss=float("inf"),dream_loss=0,
            guard_failures=guard_failures,
            loss_thresh=self.slt,dream_thresh=self.sdt,
            fail_thresh=self.sft)
        spawned=False
        if spawn and bank.n_adapters>0:
            log.info("neurogenesis triggered: %s (adapters=%d)",reason,bank.n_adapters)
            bank.freeze_all()
            bank.snapshot_adapter(model,domain,topic)
            self.rg.freeze_current(model)
            self.rg.add_rank_chunk(model)
            spawned=True
        elif bank.n_adapters==0:
            bank.snapshot_adapter(model,domain,topic)
        opt=AdamW([p for p in model.parameters() if p.requires_grad],lr=self.lr)
        col=WeightedLMCollator(tok)
        dl=make_ep_dl(ds,tok,col,self.bs,self.max_len)
        dbuf=DreamBuffer(dream_prompts,tok,self.dl,self.dn) if not dbank else None
        ddl=None
        if distill_ds is not None and len(distill_ds)>0:
            ddl=make_ep_dl(distill_ds,tok,col,self.bs,self.max_len)
        n_ep=len(ds)
        all_w=torch.tensor([ds[i].get("authority",1.0) for i in range(len(ds))])
        eps_k=evidence_eps(all_w,n_ep,self.eps_min,self.alpha)
        lam=self.lam;mu=self.mu
        warmup_steps=max(1,int(self.ms*self.pi_warmup_frac)) if self.ms>0 else 10
        tot_loss,tot_dl,tot_distl,steps=0.0,0.0,0.0,0
        hist=[]
        ddl_iter=iter(ddl) if ddl else None
        done=False
        while not done:
            for batch in dl:
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
                            t_flt={k:v.to(t_dev) for k,v in flt.items()}
                            t_out=teacher(**t_flt)
                        l_dr=multi_temp_dream_kl(s_out.logits,t_out.logits.to(dev),temps)
                    else:
                        l_dr=torch.tensor(0.0,device=dev)
                else:
                    d_inp=dbuf.sample(dev)
                    if d_inp is not None:
                        flt={k:v for k,v in d_inp.items() if k in ("input_ids","attention_mask")}
                        s_out=model(**flt)
                        with torch.no_grad():
                            t_flt={k:v.to(t_dev) for k,v in flt.items()}
                            t_out=teacher(**t_flt)
                        l_dr=dream_kl(s_out.logits,t_out.logits.to(dev),self.temp)
                    else:
                        l_dr=torch.tensor(0.0,device=dev)
                l_dist=torch.tensor(0.0,device=dev)
                if ddl_iter is not None:
                    try:
                        dbatch=next(ddl_iter)
                    except StopIteration:
                        ddl_iter=iter(ddl)
                        try:dbatch=next(ddl_iter)
                        except StopIteration:dbatch=None
                    if dbatch is not None:
                        dbatch={k:v.to(dev) if isinstance(v,torch.Tensor) else v for k,v in dbatch.items()}
                        d_ids=dbatch["input_ids"];d_mask=dbatch["attention_mask"];d_w=dbatch["weights"]
                        d_out=model(input_ids=d_ids,attention_mask=d_mask)
                        l_dist=weighted_ce(d_out.logits,d_ids,d_mask,d_w)
                l_ortho=self.rg.ortho_penalty(model).to(dev)
                loss=l_ep+lam*l_dr+mu*l_dist+l_ortho
                opt.zero_grad()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    d=l_dr.item();dd=l_dist.item()
                    if self.use_pi and steps>=warmup_steps:
                        if d<self.d_targ/1.5:lam=lam*0.9
                        elif d>1.5*self.d_targ:lam=lam*1.5
                        lam=max(self.lam_floor,min(self.lam_ceil,lam))
                    elif not self.use_pi:
                        lam=max(0.0,lam+self.rho*(d-eps_k))
                tot_loss+=l_ep.item();tot_dl+=d;tot_distl+=dd;steps+=1
                hist.append({"loss":l_ep.item(),"dream_loss":d,"distill_loss":dd,
                             "lambda":lam,"mu":mu,"ortho":l_ortho.item(),
                             "eps_k":eps_k,"d_targ":self.d_targ,"warmup":steps<warmup_steps,
                             "n_adapters":bank.n_adapters,"rank_chunks":self.rg.n_chunks,
                             "total_rank":self.rg.total_rank,"spawned":spawned})
            if self.ms<=0:break
        avg_loss=tot_loss/max(steps,1)
        avg_dl=tot_dl/max(steps,1)
        avg_distl=tot_distl/max(steps,1)
        bank.snapshot_adapter(model,domain,topic)
        return TrainResult(loss=avg_loss,steps=steps,lr=self.lr,
                           dream_loss=avg_dl,
                           extras={"lambda":lam,"mu":mu,"eps_k":eps_k,
                                   "d_targ":self.d_targ,"use_pi":self.use_pi,
                                   "distill_loss":avg_distl,
                                   "n_adapters":bank.n_adapters,
                                   "rank_chunks":self.rg.n_chunks,
                                   "total_rank":self.rg.total_rank,
                                   "spawned":spawned,"spawn_reason":reason},
                           history=hist)
