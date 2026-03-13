from __future__ import annotations
import os,logging,math,time
import torch
from wm.types import TrainResult

log=logging.getLogger(__name__)

def _make_model_input(tok,text:str,max_len:int=256):
    import tinker
    ids=tok(text,truncation=True,max_length=max_len)["input_ids"]
    return tinker.ModelInput.from_ints(ids)

def _make_datum(tok,text:str,weight:float=1.0,max_len:int=256):
    import tinker
    from tinker import TensorData
    ids=tok(text,truncation=True,max_length=max_len)["input_ids"]
    mi=tinker.ModelInput.from_ints(ids)
    tgt=torch.tensor(ids[1:]+[ids[-1]],dtype=torch.long)
    w=torch.full_like(tgt,weight,dtype=torch.float32)
    return tinker.Datum(
        model_input=mi,
        loss_fn_inputs={
            "target_tokens":TensorData.from_torch(tgt),
            "weights":TensorData.from_torch(w),
        })

class TinkerEATRDRunner:
    def __init__(self,model_name:str="Qwen/Qwen2.5-7B",
                 lora_rank:int=32,lr:float=2e-4,max_steps:int=100,
                 bs:int=4,max_len:int=256,
                 lam_init:float=1.0,d_targ:float=0.5,
                 lam_floor:float=0.01,lam_ceil:float=10.0,
                 pi_warmup_frac:float=0.15,
                 dream_n:int=4,
                 tinker_api_key:str|None=None):
        self.model_name=model_name
        self.lora_rank=lora_rank
        self.lr=lr;self.ms=max_steps;self.bs=bs;self.ml=max_len
        self.lam=lam_init;self.d_targ=d_targ
        self.lam_floor=lam_floor;self.lam_ceil=lam_ceil
        self.pi_warmup_frac=pi_warmup_frac
        self.dn=dream_n
        if tinker_api_key:
            os.environ["TINKER_API_KEY"]=tinker_api_key
        self._tc=None
        self._sc=None
        self._teacher_sc=None
        self._ckp_path=None

    def init_clients(self):
        import tinker
        log.info("tinker_eatrd: creating clients for %s rank=%d",
                 self.model_name,self.lora_rank)
        sc=tinker.ServiceClient()
        self._tc=sc.create_lora_training_client(
            base_model=self.model_name,rank=self.lora_rank)
        self._sc=self._tc.save_weights_and_get_sampling_client(
            name="eatrd_init")
        self._teacher_sc=self._tc.save_weights_and_get_sampling_client(
            name="eatrd_teacher_snap")
        log.info("tinker_eatrd: clients ready")

    def run(self,ds,dream_prompts:list[str],tok,
            dbank=None)->TrainResult:
        import tinker
        from tinker import TensorData

        if self._tc is None:
            self.init_clients()
        tc=self._tc
        teacher_sc=self._teacher_sc
        log.info("tinker_eatrd: starting training round")

        ep_texts=[]
        ep_weights=[]
        for i in range(len(ds)):
            row=ds[i]
            ep_texts.append(row.get("text",""))
            ep_weights.append(row.get("authority",1.0))

        if dbank is not None:
            import random as _rnd
            pool=[]
            for bk,prompts in dbank._b.items():
                pool.extend(prompts)
            if pool:
                dream_texts=[_rnd.choice(pool) for _ in range(self.dn*self.ms)]
            else:
                dream_texts=dream_prompts[:self.dn*self.ms]
        else:
            dream_texts=dream_prompts*((self.dn*self.ms//max(len(dream_prompts),1))+1)
            dream_texts=dream_texts[:self.dn*self.ms]

        log.info("tinker_eatrd: precomputing teacher logprobs on %d dream prompts...",
                 len(dream_texts))
        t0=time.time()
        teacher_lps=[]
        for di,dt in enumerate(dream_texts):
            mi=_make_model_input(tok,dt,self.ml)
            resp=teacher_sc.compute_logprobs(mi).result()
            lps=[x if x is not None else 0.0 for x in resp]
            teacher_lps.append(torch.tensor(lps))
            if di%50==0:
                log.info("    teacher logprobs: %d/%d",di,len(dream_texts))
        log.info("tinker_eatrd: teacher logprobs done in %.1fs",time.time()-t0)

        lam=self.lam
        warmup_steps=max(1,int(self.ms*self.pi_warmup_frac))
        tot_loss,tot_dl,steps=0.0,0.0,0
        hist=[]
        dream_idx=0
        adam=tinker.AdamParams(learning_rate=self.lr,beta1=0.9,beta2=0.95,eps=1e-8)

        log.info("tinker_eatrd: training %d steps, bs=%d, lam_init=%.2f, warmup=%d",
                 self.ms,self.bs,self.lam,warmup_steps)

        for step in range(self.ms):
            ep_batch=[]
            for _ in range(self.bs):
                idx=step*self.bs+_
                idx=idx%len(ep_texts)
                ep_batch.append(_make_datum(tok,ep_texts[idx],ep_weights[idx],self.ml))

            fwd_res=tc.forward_backward(ep_batch,loss_fn="cross_entropy").result()
            ep_loss_vals=[]
            for out in fwd_res.loss_fn_outputs:
                lp=out["logprobs"].to_torch()
                ep_loss_vals.append(-lp.mean().item())
            l_ep=sum(ep_loss_vals)/max(len(ep_loss_vals),1)

            d_batch_texts=dream_texts[dream_idx:dream_idx+self.dn]
            d_batch_teacher=teacher_lps[dream_idx:dream_idx+self.dn]
            dream_idx+=self.dn
            if dream_idx>=len(dream_texts):dream_idx=0

            kl_vals=[]
            if d_batch_texts:
                d_data=[]
                for dt in d_batch_texts:
                    d_data.append(_make_datum(tok,dt,1.0,self.ml))

                def dream_kl_loss(data,logprobs):
                    total_kl=torch.tensor(0.0)
                    for i,lp in enumerate(logprobs):
                        if i<len(d_batch_teacher):
                            tlp=d_batch_teacher[i]
                            mn=min(len(lp),len(tlp))
                            kl=(lp[:mn]-tlp[:mn]).mean()
                            total_kl=total_kl+kl
                            kl_vals.append(kl.item())
                    return total_kl,{"kl":total_kl.item()}

                tc.forward_backward_custom(d_data,dream_kl_loss).result()
                l_dr=sum(kl_vals)/max(len(kl_vals),1) if kl_vals else 0.0
            else:
                l_dr=0.0

            tc.optim_step(adam).result()

            if step>=warmup_steps:
                if l_dr<self.d_targ/1.5:
                    lam=lam*0.9
                elif l_dr>1.5*self.d_targ:
                    lam=lam*1.5
                lam=max(self.lam_floor,min(self.lam_ceil,lam))

            tot_loss+=l_ep;tot_dl+=abs(l_dr);steps+=1
            hist.append({"loss":l_ep,"dream_loss":l_dr,"lambda":lam,
                         "d_targ":self.d_targ,"warmup":step<warmup_steps})

            if step%10==0 or step==self.ms-1:
                log.info("  step %d/%d: ep_loss=%.4f dream_kl=%.4f lam=%.4f%s",
                         step+1,self.ms,l_ep,l_dr,lam,
                         " [warmup]" if step<warmup_steps else "")

        log.info("tinker_eatrd: training complete, saving checkpoint")
        ckp_path=tc.save_state(name=f"eatrd_step{steps}").result().path
        self._sc=tc.save_weights_and_get_sampling_client(name=f"eatrd_post_{steps}")
        log.info("tinker_eatrd: checkpoint at %s, sampling client updated",ckp_path)
        self._ckp_path=ckp_path

        avg_loss=tot_loss/max(steps,1)
        avg_dl=tot_dl/max(steps,1)
        return TrainResult(loss=avg_loss,steps=steps,lr=self.lr,
                           dream_loss=avg_dl,
                           extras={"lambda":lam,"d_targ":self.d_targ,
                                   "backend":"tinker","model":self.model_name,
                                   "lora_rank":self.lora_rank,
                                   "checkpoint":ckp_path},
                           history=hist)

    def get_sampling_client(self,name:str="eatrd_sampler"):
        if self._tc is not None:
            self._sc=self._tc.save_weights_and_get_sampling_client(name=name)
        return self._sc

    def sample(self,prompt:str,tok,max_tokens:int=256,temp:float=0.7,
               timeout:float=60.0)->str:
        import tinker
        if self._sc is None:
            if self._tc is None:
                self.init_clients()
            else:
                self._sc=self._tc.save_weights_and_get_sampling_client(name="eatrd_sample")
        mi=_make_model_input(tok,prompt,self.ml)
        sp=tinker.SamplingParams(max_tokens=max_tokens,temperature=max(temp,0.01),top_p=0.95)
        try:
            resp=self._sc.sample(mi,num_samples=1,sampling_params=sp).result(timeout=timeout)
            if resp.sequences:
                return tok.decode(resp.sequences[0].tokens,skip_special_tokens=True)
        except Exception as e:
            log.warning("tinker sample failed: %s",e)
        return ""
