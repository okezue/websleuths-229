from __future__ import annotations
import logging,os,json,math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass,field
from typing import Optional

log=logging.getLogger(__name__)

@dataclass
class NeuronNode:
    layer:int
    position:int
    neuron_idx:int
    activation:float=0.0
    node_type:str="mlp"

@dataclass
class CircuitEdge:
    src:int
    tgt:int
    weight:float=0.0

@dataclass
class NeuronGraph:
    prompt:str=""
    tokens:list[str]=field(default_factory=list)
    nodes:list[dict]=field(default_factory=list)
    adjacency:Optional[torch.Tensor]=None
    logit_targets:list[dict]=field(default_factory=list)
    n_layers:int=0
    n_positions:int=0
    def save(self,path:str):
        data={"prompt":self.prompt,"tokens":self.tokens,"nodes":self.nodes,
              "logit_targets":self.logit_targets,"n_layers":self.n_layers,
              "n_positions":self.n_positions}
        if self.adjacency is not None:
            data["adjacency"]=self.adjacency.cpu().tolist()
        with open(path,"w") as f:
            json.dump(data,f)
    @classmethod
    def load(cls,path:str)->"NeuronGraph":
        with open(path) as f:
            data=json.load(f)
        g=cls(prompt=data["prompt"],tokens=data.get("tokens",[]),
              nodes=data.get("nodes",[]),logit_targets=data.get("logit_targets",[]),
              n_layers=data.get("n_layers",0),n_positions=data.get("n_positions",0))
        if "adjacency" in data:
            g.adjacency=torch.tensor(data["adjacency"])
        return g

class NeuronTracer:
    def __init__(self,model,tok,top_k_neurons:int=50,dev=None):
        self._m=model
        self._t=tok
        self._k=top_k_neurons
        self._dev=dev or next(model.parameters()).device
        self._hooks=[]
        self._mlp_ins={}
        self._mlp_outs={}
        self._mlp_acts={}
        self._attn_pats={}
    def _find_mlp_modules(self)->list[tuple[int,nn.Module]]:
        mlps=[]
        for name,mod in self._m.named_modules():
            if any(k in name for k in [".mlp",".feed_forward"]) and not any(k in name for k in [".mlp.",".feed_forward."]):
                try:
                    li=int([p for p in name.split(".") if p.isdigit()][0])
                    mlps.append((li,mod))
                except (IndexError,ValueError):
                    pass
        mlps.sort(key=lambda x:x[0])
        return mlps
    def _register_hooks(self,mlps):
        self._clear_hooks()
        for li,mod in mlps:
            def make_hook(layer_idx):
                def hook_fn(module,inp,out):
                    if isinstance(inp,tuple):inp=inp[0]
                    self._mlp_ins[layer_idx]=inp.detach()
                    if isinstance(out,tuple):out=out[0]
                    self._mlp_outs[layer_idx]=out.detach()
                return hook_fn
            h=mod.register_forward_hook(make_hook(li))
            self._hooks.append(h)
    def _clear_hooks(self):
        for h in self._hooks:h.remove()
        self._hooks=[]
        self._mlp_ins.clear()
        self._mlp_outs.clear()
    def _get_mlp_weights(self,mlp_mod)->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        gate=up=down=None
        for n,p in mlp_mod.named_parameters():
            if "gate" in n and "weight" in n:gate=p.data
            elif "up" in n and "weight" in n:up=p.data
            elif "down" in n and "weight" in n:down=p.data
        if gate is None and up is None:
            params=list(mlp_mod.parameters())
            if len(params)>=2:
                up=params[0].data
                down=params[-1].data
        return gate,up,down
    def trace(self,prompt:str,max_logits:int=5)->NeuronGraph:
        self._m.eval()
        mlps=self._find_mlp_modules()
        if not mlps:
            log.warning("no MLP modules found")
            return NeuronGraph(prompt=prompt)
        self._register_hooks(mlps)
        enc=self._t(prompt,return_tensors="pt",truncation=True,max_length=256)
        inp={k:v.to(self._dev) for k,v in enc.items() if k in ("input_ids","attention_mask")}
        tokens=[self._t.decode([t]) for t in inp["input_ids"][0]]
        with torch.no_grad():
            out=self._m(**inp)
        logits=out.logits[0,-1,:]
        probs=F.softmax(logits,dim=-1)
        topk=torch.topk(probs,max_logits)
        logit_targets=[{"token":self._t.decode([idx.item()]),"prob":p.item(),
                        "idx":idx.item()} for idx,p in zip(topk.indices,topk.values)]
        nodes=[]
        all_active=[]
        for li,mod in mlps:
            if li not in self._mlp_outs:continue
            mlp_out=self._mlp_outs[li]
            norms=mlp_out[0].norm(dim=-1)
            for pos in range(mlp_out.shape[1]):
                gate,up,down=self._get_mlp_weights(mod)
                if down is not None:
                    out_vec=mlp_out[0,pos,:]
                    try:
                        neuron_contrib=torch.matmul(down,out_vec)
                    except RuntimeError:
                        try:
                            neuron_contrib=torch.matmul(down.T,out_vec)
                        except RuntimeError:
                            neuron_contrib=out_vec
                    topn=torch.topk(neuron_contrib.abs(),min(self._k,len(neuron_contrib)))
                    for ni,act in zip(topn.indices,topn.values):
                        all_active.append({"layer":li,"position":pos,
                            "neuron":ni.item(),"activation":act.item(),
                            "type":"mlp"})
        all_active.sort(key=lambda x:abs(x["activation"]),reverse=True)
        nodes=all_active[:self._k*len(mlps)]
        for i,tok_str in enumerate(tokens):
            nodes.append({"layer":-1,"position":i,"neuron":-1,
                          "activation":0,"type":"token","token":tok_str})
        for lt in logit_targets:
            nodes.append({"layer":len(mlps),"position":len(tokens)-1,
                          "neuron":lt["idx"],"activation":lt["prob"],
                          "type":"logit","token":lt["token"]})
        n=len(nodes)
        adj=torch.zeros(n,n)
        mlp_nodes=[i for i,nd in enumerate(nodes) if nd["type"]=="mlp"]
        tok_nodes=[i for i,nd in enumerate(nodes) if nd["type"]=="token"]
        log_nodes=[i for i,nd in enumerate(nodes) if nd["type"]=="logit"]
        for i in mlp_nodes:
            for j in tok_nodes:
                if nodes[j]["position"]<=nodes[i]["position"]:
                    adj[i,j]=abs(nodes[i]["activation"])*0.1
        for i in mlp_nodes:
            for j in mlp_nodes:
                if i!=j and nodes[j]["layer"]<nodes[i]["layer"]:
                    if nodes[j]["position"]<=nodes[i]["position"]:
                        li=nodes[i]["layer"];lj=nodes[j]["layer"]
                        if li not in self._mlp_outs or lj not in self._mlp_outs:continue
                        oi=self._mlp_outs[li][0,nodes[i]["position"],:]
                        oj=self._mlp_outs[lj][0,nodes[j]["position"],:]
                        cos=F.cosine_similarity(oi.unsqueeze(0),oj.unsqueeze(0)).item()
                        adj[i,j]=abs(cos)*abs(nodes[j]["activation"])
        for i in log_nodes:
            for j in mlp_nodes:
                last_layer=max(nd["layer"] for nd in nodes if nd["type"]=="mlp")
                if nodes[j]["layer"]==last_layer:
                    adj[i,j]=abs(nodes[j]["activation"])
        self._clear_hooks()
        g=NeuronGraph(prompt=prompt,tokens=tokens,nodes=nodes,adjacency=adj,
                       logit_targets=logit_targets,n_layers=len(mlps),
                       n_positions=len(tokens))
        return g
    def trace_and_compare(self,prompt:str,base_model,max_logits:int=5)->dict:
        g1=self.trace(prompt,max_logits)
        base_tracer=NeuronTracer(base_model,self._t,self._k,
                                  next(base_model.parameters()).device)
        g2=base_tracer.trace(prompt,max_logits)
        trained_acts={(n["layer"],n["position"],n["neuron"]):n["activation"]
                       for n in g1.nodes if n["type"]=="mlp"}
        base_acts={(n["layer"],n["position"],n["neuron"]):n["activation"]
                    for n in g2.nodes if n["type"]=="mlp"}
        all_keys=set(trained_acts.keys())|set(base_acts.keys())
        diffs=[]
        for k in all_keys:
            ta=trained_acts.get(k,0)
            ba=base_acts.get(k,0)
            if abs(ta-ba)>0.01:
                diffs.append({"layer":k[0],"position":k[1],"neuron":k[2],
                              "trained":ta,"base":ba,"delta":ta-ba})
        diffs.sort(key=lambda x:abs(x["delta"]),reverse=True)
        return {"prompt":prompt,"n_changed":len(diffs),"top_changes":diffs[:20],
                "trained_logits":g1.logit_targets,"base_logits":g2.logit_targets,
                "trained_graph":g1,"base_graph":g2}
    def plot_graph(self,graph:NeuronGraph,path:str,title:str=""):
        try:
            import matplotlib;matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except:return
        mlp_nodes=[(i,n) for i,n in enumerate(graph.nodes) if n["type"]=="mlp"]
        if not mlp_nodes:return
        fig,axes=plt.subplots(2,2,figsize=(16,12))
        layers=[n["layer"] for _,n in mlp_nodes]
        acts=[abs(n["activation"]) for _,n in mlp_nodes]
        axes[0,0].scatter(layers,acts,c=layers,cmap="viridis",alpha=0.6,s=20)
        axes[0,0].set_xlabel("Layer");axes[0,0].set_ylabel("|Activation|")
        axes[0,0].set_title("Neuron Activations by Layer")
        layer_counts={}
        for _,n in mlp_nodes:
            l=n["layer"]
            layer_counts[l]=layer_counts.get(l,0)+1
        ls=sorted(layer_counts.keys())
        axes[0,1].bar(ls,[layer_counts[l] for l in ls],color="#4CAF50",alpha=0.7)
        axes[0,1].set_xlabel("Layer");axes[0,1].set_ylabel("Active Neurons")
        axes[0,1].set_title("Active Neuron Count per Layer")
        if graph.adjacency is not None:
            adj=graph.adjacency.numpy()
            n=min(adj.shape[0],50)
            im=axes[1,0].imshow(adj[:n,:n],cmap="RdBu_r",aspect="auto",
                                 vmin=-adj[:n,:n].max(),vmax=adj[:n,:n].max())
            axes[1,0].set_title(f"Adjacency Matrix (top {n} nodes)")
            plt.colorbar(im,ax=axes[1,0])
        if graph.logit_targets:
            toks=[lt["token"] for lt in graph.logit_targets]
            probs=[lt["prob"] for lt in graph.logit_targets]
            axes[1,1].barh(range(len(toks)),probs,color="#2196F3")
            axes[1,1].set_yticks(range(len(toks)));axes[1,1].set_yticklabels(toks)
            axes[1,1].set_xlabel("Probability");axes[1,1].set_title("Top Logit Predictions")
        fig.suptitle(title or f"Circuit: {graph.prompt[:60]}",fontsize=13)
        plt.tight_layout()
        plt.savefig(path,dpi=150)
        plt.close()
    def plot_comparison(self,result:dict,path:str):
        try:
            import matplotlib;matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except:return
        diffs=result.get("top_changes",[])
        if not diffs:return
        fig,axes=plt.subplots(1,3,figsize=(18,6))
        layers=[d["layer"] for d in diffs]
        deltas=[d["delta"] for d in diffs]
        colors=["#F44336" if d>0 else "#2196F3" for d in deltas]
        axes[0].barh(range(len(diffs)),deltas,color=colors)
        axes[0].set_yticks(range(len(diffs)))
        axes[0].set_yticklabels([f"L{d['layer']}N{d['neuron']}" for d in diffs],fontsize=7)
        axes[0].set_xlabel("Activation Delta (trained - base)")
        axes[0].set_title(f"Top {len(diffs)} Changed Neurons")
        layer_deltas={}
        for d in diffs:
            l=d["layer"]
            layer_deltas[l]=layer_deltas.get(l,0)+abs(d["delta"])
        ls=sorted(layer_deltas.keys())
        axes[1].bar(ls,[layer_deltas[l] for l in ls],color="#9C27B0",alpha=0.7)
        axes[1].set_xlabel("Layer");axes[1].set_ylabel("Total |Delta|")
        axes[1].set_title("Change Magnitude by Layer")
        tl=result.get("trained_logits",[])
        bl=result.get("base_logits",[])
        if tl and bl:
            x=np.arange(len(tl))
            w=0.35
            axes[2].bar(x-w/2,[l["prob"] for l in bl],w,label="Base",color="#FF9800",alpha=0.7)
            axes[2].bar(x+w/2,[l["prob"] for l in tl],w,label="Trained",color="#4CAF50",alpha=0.7)
            axes[2].set_xticks(x)
            axes[2].set_xticklabels([l["token"] for l in tl],fontsize=8)
            axes[2].set_ylabel("Probability");axes[2].set_title("Logit Shift")
            axes[2].legend()
        fig.suptitle(f"Circuit Comparison: {result['prompt'][:60]}",fontsize=13)
        plt.tight_layout()
        plt.savefig(path,dpi=150)
        plt.close()
