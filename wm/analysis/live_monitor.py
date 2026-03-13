from __future__ import annotations
import logging,os,json,math
from dataclasses import dataclass,field
from collections import defaultdict

log=logging.getLogger(__name__)

@dataclass
class DomainSnapshot:
    domain:str=""
    topic:str=""
    step:int=0
    loss:float=0.0
    dream_loss:float=0.0
    distill_loss:float=0.0
    lam:float=0.0
    mu:float=0.0
    n_adapters:int=0
    rank_chunks:int=0
    total_rank:int=0
    spawned:bool=False
    mmlu:dict=field(default_factory=dict)
    bench:dict=field(default_factory=dict)
    anchor_nll:float=0.0
    grad_norm:float=0.0
    param_norm:float=0.0
    lora_rank_util:float=0.0
    weight_delta_norm:float=0.0
    kl_from_base:float=0.0

class LiveMonitor:
    def __init__(self,out_dir:str,baseline_state:dict|None=None):
        self._od=out_dir
        os.makedirs(f"{out_dir}/analysis",exist_ok=True)
        self._snaps:list[DomainSnapshot]=[]
        self._base_state=baseline_state
        self._domain_losses=defaultdict(list)
        self._forgetting=defaultdict(list)
        self._grad_norms=[]
        self._weight_deltas=[]
        self._neurogenesis_events=[]
        self._per_domain_benchmarks=defaultdict(list)
    def record_snapshot(self,snap:DomainSnapshot):
        self._snaps.append(snap)
        self._domain_losses[snap.domain].append({"step":snap.step,"loss":snap.loss,
            "dream":snap.dream_loss,"distill":snap.distill_loss})
        if snap.grad_norm>0:self._grad_norms.append({"step":snap.step,"norm":snap.grad_norm,"domain":snap.domain})
        if snap.weight_delta_norm>0:self._weight_deltas.append({"step":snap.step,"delta":snap.weight_delta_norm,"domain":snap.domain})
        if snap.spawned:self._neurogenesis_events.append({"step":snap.step,"domain":snap.domain,"topic":snap.topic,"n_adapters":snap.n_adapters,"rank":snap.total_rank})
    def record_forgetting(self,domain:str,bench_name:str,baseline:float,current:float,step:int):
        self._forgetting[domain].append({"bench":bench_name,"baseline":baseline,"current":current,"delta":current-baseline,"step":step})
        self._per_domain_benchmarks[domain].append({"bench":bench_name,"acc":current,"step":step})
    def compute_model_stats(self,model,base_state:dict|None=None)->dict:
        import torch
        stats={}
        gn=0.0;pn=0.0;n=0
        for p in model.parameters():
            if p.requires_grad:
                pn+=p.data.norm().item()**2
                if p.grad is not None:
                    gn+=p.grad.norm().item()**2
                n+=1
        stats["grad_norm"]=math.sqrt(gn) if gn>0 else 0
        stats["param_norm"]=math.sqrt(pn)
        stats["n_trainable"]=n
        if base_state:
            delta=0.0
            for k,v in model.named_parameters():
                if k in base_state and v.requires_grad:
                    delta+=(v.data.cpu()-base_state[k]).norm().item()**2
            stats["weight_delta"]=math.sqrt(delta)
        mem=0.0
        if torch.cuda.is_available():
            mem=torch.cuda.max_memory_allocated()/1e9
        stats["gpu_mem_gb"]=mem
        return stats
    def compute_kl_from_base(self,model,base_model,tok,prompts:list[str])->float:
        import torch,torch.nn.functional as F
        dev=next(model.parameters()).device
        kls=[]
        model.eval();base_model.eval()
        for p in prompts[:5]:
            enc=tok(p,return_tensors="pt",truncation=True,max_length=128)
            inp={k:v.to(dev) for k,v in enc.items() if k in ("input_ids","attention_mask")}
            with torch.no_grad():
                cur=model(**inp).logits
                base=base_model(**inp).logits
            kl=F.kl_div(F.log_softmax(cur[:,-1,:],dim=-1),F.softmax(base[:,-1,:],dim=-1),reduction="batchmean")
            kls.append(kl.item())
        return sum(kls)/max(len(kls),1) if kls else 0.0
    def save_analysis(self):
        data={"snapshots":[{"domain":s.domain,"topic":s.topic,"step":s.step,"loss":s.loss,
            "dream_loss":s.dream_loss,"distill_loss":s.distill_loss,"lam":s.lam,"mu":s.mu,
            "n_adapters":s.n_adapters,"rank_chunks":s.rank_chunks,"total_rank":s.total_rank,
            "spawned":s.spawned,"anchor_nll":s.anchor_nll,"grad_norm":s.grad_norm,
            "param_norm":s.param_norm,"weight_delta":s.weight_delta_norm,
            "kl_from_base":s.kl_from_base} for s in self._snaps],
            "domain_losses":dict(self._domain_losses),
            "forgetting":dict(self._forgetting),
            "grad_norms":self._grad_norms,
            "weight_deltas":self._weight_deltas,
            "neurogenesis_events":self._neurogenesis_events,
            "per_domain_benchmarks":dict(self._per_domain_benchmarks)}
        with open(f"{self._od}/analysis/live_data.json","w") as f:
            json.dump(data,f,indent=2,default=str)
    def plot_all(self):
        try:
            self._plot_forgetting_heatmap()
            self._plot_loss_landscape()
            self._plot_neurogenesis_timeline()
            self._plot_grad_flow()
            self._plot_weight_drift()
            self._plot_capacity_util()
            self._plot_kl_divergence()
            self._plot_domain_radar()
        except Exception as e:
            log.warning("plot_all failed: %s",e)
    def _plot_forgetting_heatmap(self):
        import matplotlib;matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        if not self._forgetting:return
        domains=list(self._forgetting.keys())
        benches=sorted({e["bench"] for v in self._forgetting.values() for e in v})
        if not benches:return
        mx=np.zeros((len(domains),len(benches)))
        for di,d in enumerate(domains):
            for e in self._forgetting[d]:
                if e["bench"] in benches:
                    bi=benches.index(e["bench"])
                    mx[di,bi]=e["delta"]
        fig,ax=plt.subplots(figsize=(10,6))
        im=ax.imshow(mx,cmap="RdYlGn",aspect="auto",vmin=-0.15,vmax=0.15)
        ax.set_xticks(range(len(benches)));ax.set_xticklabels(benches,rotation=45,ha="right",fontsize=8)
        ax.set_yticks(range(len(domains)));ax.set_yticklabels(domains)
        for i in range(len(domains)):
            for j in range(len(benches)):
                ax.text(j,i,f"{mx[i,j]:+.3f}",ha="center",va="center",fontsize=7,
                        color="white" if abs(mx[i,j])>0.08 else "black")
        plt.colorbar(im,label="Accuracy Delta vs Baseline")
        ax.set_title("Forgetting Heatmap: Domain x Benchmark")
        plt.tight_layout()
        plt.savefig(f"{self._od}/analysis/forgetting_heatmap.png",dpi=150)
        plt.close()
    def _plot_loss_landscape(self):
        import matplotlib;matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not self._snaps:return
        fig,axes=plt.subplots(3,1,figsize=(14,10),sharex=True)
        colors={"finance":"#2196F3","legal":"#FF9800","chemistry":"#4CAF50","medicine":"#E91E63"}
        for d,entries in self._domain_losses.items():
            steps=[e["step"] for e in entries]
            c=colors.get(d,"gray")
            axes[0].plot(steps,[e["loss"] for e in entries],label=d,color=c,alpha=0.8)
            axes[1].plot(steps,[e["dream"] for e in entries],label=d,color=c,alpha=0.8)
            axes[2].plot(steps,[e["distill"] for e in entries],label=d,color=c,alpha=0.8)
        for ax,title in zip(axes,["Episode Loss (L_ep)","Dream Loss (L_dream)","Distill Loss (L_distill)"]):
            ax.set_ylabel(title);ax.legend(fontsize=7);ax.grid(True,alpha=0.3)
        axes[-1].set_xlabel("Global Step")
        fig.suptitle("Three-Loss Landscape Across Domains",fontsize=13)
        plt.tight_layout()
        plt.savefig(f"{self._od}/analysis/loss_landscape.png",dpi=150)
        plt.close()
    def _plot_neurogenesis_timeline(self):
        import matplotlib;matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not self._snaps:return
        fig,axes=plt.subplots(3,1,figsize=(14,8),sharex=True)
        steps=[s.step for s in self._snaps]
        axes[0].plot(steps,[s.n_adapters for s in self._snaps],color="#673AB7",linewidth=2)
        axes[0].set_ylabel("Active Adapters")
        for ev in self._neurogenesis_events:
            axes[0].axvline(ev["step"],color="red",linestyle="--",alpha=0.5)
            axes[0].annotate(f"spawn\n{ev['domain'][:3]}",xy=(ev["step"],ev["n_adapters"]),fontsize=6)
        axes[1].plot(steps,[s.total_rank for s in self._snaps],color="#009688",linewidth=2)
        axes[1].set_ylabel("Total LoRA Rank")
        axes[2].plot(steps,[s.lam for s in self._snaps],label="λ (dream)",color="#FF5722")
        axes[2].plot(steps,[s.mu for s in self._snaps],label="μ (distill)",color="#3F51B5")
        axes[2].set_ylabel("Loss Weights");axes[2].legend(fontsize=8)
        axes[-1].set_xlabel("Global Step")
        fig.suptitle("Neurogenesis: Capacity Growth & Loss Weight Evolution",fontsize=13)
        for ax in axes:ax.grid(True,alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self._od}/analysis/neurogenesis_timeline.png",dpi=150)
        plt.close()
    def _plot_grad_flow(self):
        import matplotlib;matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not self._grad_norms:return
        fig,ax=plt.subplots(figsize=(12,5))
        colors={"finance":"#2196F3","legal":"#FF9800","chemistry":"#4CAF50","medicine":"#E91E63"}
        for g in self._grad_norms:
            ax.scatter(g["step"],g["norm"],c=colors.get(g["domain"],"gray"),s=8,alpha=0.6)
        ax.set_xlabel("Step");ax.set_ylabel("Gradient L2 Norm")
        ax.set_title("Gradient Flow: Norm per Training Step (color=domain)")
        ax.grid(True,alpha=0.3)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=c,label=d) for d,c in colors.items()],fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{self._od}/analysis/grad_flow.png",dpi=150)
        plt.close()
    def _plot_weight_drift(self):
        import matplotlib;matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not self._weight_deltas:return
        fig,ax=plt.subplots(figsize=(12,5))
        steps=[w["step"] for w in self._weight_deltas]
        deltas=[w["delta"] for w in self._weight_deltas]
        ax.plot(steps,deltas,color="#795548",linewidth=1.5)
        ax.fill_between(steps,deltas,alpha=0.2,color="#795548")
        ax.set_xlabel("Step");ax.set_ylabel("||θ - θ_base||₂")
        ax.set_title("Weight Drift from Baseline (L2 distance)")
        ax.grid(True,alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self._od}/analysis/weight_drift.png",dpi=150)
        plt.close()
    def _plot_capacity_util(self):
        import matplotlib;matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not self._snaps:return
        fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
        steps=[s.step for s in self._snaps]
        ranks=[s.total_rank for s in self._snaps]
        adapters=[s.n_adapters for s in self._snaps]
        axes[0].fill_between(steps,ranks,alpha=0.4,color="#00BCD4")
        axes[0].plot(steps,ranks,color="#00BCD4",linewidth=2)
        axes[0].set_ylabel("Total Rank (capacity)")
        axes[0].set_title("Capacity Utilization Over Training")
        axes[1].bar(steps,adapters,width=max(1,len(steps)//100),color="#9C27B0",alpha=0.7)
        axes[1].set_ylabel("# Frozen Adapters");axes[1].set_xlabel("Step")
        for ax in axes:ax.grid(True,alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self._od}/analysis/capacity_util.png",dpi=150)
        plt.close()
    def _plot_kl_divergence(self):
        import matplotlib;matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        kls=[s.kl_from_base for s in self._snaps if s.kl_from_base>0]
        if not kls:return
        steps=[s.step for s in self._snaps if s.kl_from_base>0]
        fig,ax=plt.subplots(figsize=(12,5))
        ax.plot(steps,kls,color="#F44336",linewidth=1.5)
        ax.fill_between(steps,kls,alpha=0.15,color="#F44336")
        ax.set_xlabel("Step");ax.set_ylabel("KL(current || baseline)")
        ax.set_title("KL Divergence from Baseline Model")
        ax.grid(True,alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self._od}/analysis/kl_divergence.png",dpi=150)
        plt.close()
    def _plot_domain_radar(self):
        import matplotlib;matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        if not self._per_domain_benchmarks:return
        domains=list(self._per_domain_benchmarks.keys())
        if len(domains)<3:return
        latest={}
        for d in domains:
            entries=self._per_domain_benchmarks[d]
            if entries:latest[d]=entries[-1]["acc"]
        if len(latest)<3:return
        labels=list(latest.keys())
        vals=[latest[l] for l in labels]
        N=len(labels)
        angles=np.linspace(0,2*np.pi,N,endpoint=False).tolist()
        vals+=vals[:1];angles+=angles[:1]
        fig,ax=plt.subplots(figsize=(8,8),subplot_kw=dict(polar=True))
        ax.fill(angles,vals,alpha=0.25,color="#2196F3")
        ax.plot(angles,vals,color="#2196F3",linewidth=2)
        ax.set_xticks(angles[:-1]);ax.set_xticklabels(labels,fontsize=9)
        ax.set_title("Domain Performance Radar",fontsize=13,pad=20)
        plt.tight_layout()
        plt.savefig(f"{self._od}/analysis/domain_radar.png",dpi=150)
        plt.close()
