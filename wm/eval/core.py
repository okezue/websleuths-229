from __future__ import annotations
import math,torch
from datasets import Dataset
from wm.types import EvalReport,Probe

class Evaluator:
    def __init__(self,model,tok,max_len:int=512):
        self._m=model.cpu()
        self._t=tok
        self._ml=max_len
        self._dev=torch.device("cpu")
    def _ppl(self,ds:Dataset)->float:
        self._m.eval()
        tot,n=0.0,0
        with torch.no_grad():
            for row in ds:
                txt=row.get("text","")
                if not txt:continue
                enc={k:v.to(self._dev) for k,v in self._t(txt,return_tensors="pt",truncation=True,max_length=self._ml).items()}
                out=self._m(**enc,labels=enc["input_ids"])
                tot+=out.loss.item()
                n+=1
        return math.exp(tot/max(n,1))
    def _acc(self,ds:Dataset)->float:
        self._m.eval()
        cor,tot=0,0
        with torch.no_grad():
            for row in ds:
                txt=row.get("text","")
                if not txt:continue
                enc={k:v.to(self._dev) for k,v in self._t(txt,return_tensors="pt",truncation=True,max_length=self._ml).items()}
                ids=enc["input_ids"]
                if ids.shape[1]<2:continue
                out=self._m(**enc)
                preds=out.logits[:,:-1].argmax(dim=-1)
                tgts=ids[:,1:]
                cor+=(preds==tgts).sum().item()
                tot+=tgts.numel()
        return cor/max(tot,1)
    def _qa_f1(self,ds:Dataset)->float:
        hits,tot=0,0
        self._m.eval()
        for row in ds:
            if "prompt" not in row or "completion" not in row:continue
            enc=self._t(row["prompt"],return_tensors="pt",truncation=True,max_length=self._ml)
            inp={k:v.to(self._dev) for k,v in enc.items() if k in ("input_ids","attention_mask")}
            with torch.no_grad():
                out=self._m.generate(**inp,max_new_tokens=64,do_sample=False)
            gen=self._t.decode(out[0][inp["input_ids"].shape[1]:],skip_special_tokens=True)
            if row["completion"].lower() in gen.lower():
                hits+=1
            tot+=1
        return hits/max(tot,1)
    def _probe_score(self,probe:Probe)->float:
        enc=self._t(probe.prompt,return_tensors="pt",truncation=True,max_length=self._ml)
        inp={k:v.to(self._dev) for k,v in enc.items() if k in ("input_ids","attention_mask")}
        with torch.no_grad():
            out=self._m.generate(**inp,max_new_tokens=64,do_sample=False)
        gen=self._t.decode(out[0][inp["input_ids"].shape[1]:],skip_special_tokens=True)
        gl=probe.gold.lower().split()
        pl=gen.lower().split()
        if not gl:return 0.0
        hit=int(probe.gold.lower() in gen.lower())
        common=set(gl)&set(pl)
        prec=len(common)/max(len(pl),1)
        rec=len(common)/max(len(gl),1)
        f1=(2*prec*rec/(prec+rec)) if (prec+rec)>0 else 0.0
        return max(hit,f1)
    def evaluate_probes(self,probes:list[Probe])->dict[str,float]:
        if not probes:return {"hit_rate":0.0,"mean_f1":0.0,"n_probes":0}
        self._m.eval()
        scores=[self._probe_score(p) for p in probes]
        hits=sum(1 for s in scores if s>=0.5)
        return {
            "hit_rate":hits/len(probes),
            "mean_f1":sum(scores)/len(scores),
            "n_probes":len(probes),
        }
    def evaluate(self,ds:Dataset,do_qa:bool=False)->EvalReport:
        ppl=self._ppl(ds)
        acc=self._acc(ds)
        qa=self._qa_f1(ds) if do_qa and "prompt" in ds.column_names else 0.0
        return EvalReport(ppl=ppl,acc=acc,qa_f1=qa)
