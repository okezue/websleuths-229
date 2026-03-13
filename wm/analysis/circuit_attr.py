from __future__ import annotations
import logging,os,json,math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass,field
from typing import Optional

log=logging.getLogger(__name__)

@dataclass
class AttrGraph:
    prompt:str=""
    tokens:list[str]=field(default_factory=list)
    n_layers:int=0
    n_pos:int=0
    node_info:list[dict]=field(default_factory=list)
    adjacency:Optional[torch.Tensor]=None
    logit_targets:list[dict]=field(default_factory=list)
    node_influence:Optional[torch.Tensor]=None
    def save(self,path:str):
        d={"prompt":self.prompt,"tokens":self.tokens,"n_layers":self.n_layers,
           "n_pos":self.n_pos,"node_info":self.node_info,
           "logit_targets":self.logit_targets}
        if self.adjacency is not None:d["adjacency_shape"]=list(self.adjacency.shape)
        torch.save({"graph_meta":d,
                     "adjacency":self.adjacency,
                     "node_influence":self.node_influence},path)
    @classmethod
    def load(cls,path:str)->"AttrGraph":
        d=torch.load(path,map_location="cpu",weights_only=False)
        m=d["graph_meta"]
        g=cls(**{k:v for k,v in m.items() if k in cls.__dataclass_fields__})
        g.adjacency=d.get("adjacency")
        g.node_influence=d.get("node_influence")
        return g

class ModelCircuitTracer:
    def __init__(self,model,tok,top_k:int=100,dev=None):
        self._m=model
        self._t=tok
        self._k=top_k
        self._dev=dev or next(model.parameters()).device
        self._mlp_mods=[]
        self._hooks=[]
        self._mlp_in={}
        self._mlp_out={}
        self._resid={}
        self._find_modules()
    def _find_modules(self):
        self._mlp_mods=[]
        self._attn_mods=[]
        self._ln_mods=[]
        self._layers=[]
        for name,mod in self._m.named_modules():
            parts=name.split(".")
            digits=[p for p in parts if p.isdigit()]
            if not digits:continue
            li=int(digits[0])
            if any(k in name for k in [".mlp",".feed_forward"]):
                is_leaf=not any(k in name for k in [".mlp.",".feed_forward."])
                if is_leaf:
                    self._mlp_mods.append((li,name,mod))
                    if li not in self._layers:self._layers.append(li)
            if any(k in name for k in [".self_attn",".attention"]):
                is_leaf=not any("." in name[name.index("attn")+4:] if "attn" in name else False for _ in [0])
                self._attn_mods.append((li,name,mod))
        self._layers.sort()
        self._n_layers=len(self._layers)
        log.info("found %d MLP layers, %d attn modules",len(self._mlp_mods),len(self._attn_mods))
    def _get_down_proj(self,mlp_mod)->Optional[torch.Tensor]:
        for n,p in mlp_mod.named_parameters():
            if "down" in n and "weight" in n:
                return p.data
        params=list(mlp_mod.parameters())
        if len(params)>=2:
            return params[-1].data
        return None
    def _get_gate_proj(self,mlp_mod)->Optional[torch.Tensor]:
        for n,p in mlp_mod.named_parameters():
            if ("gate" in n or "up" in n) and "weight" in n:
                return p.data
        params=list(mlp_mod.parameters())
        if params:
            return params[0].data
        return None
    def _setup_hooks(self):
        self._clear_hooks()
        self._mlp_in.clear();self._mlp_out.clear();self._resid.clear()
        for li,name,mod in self._mlp_mods:
            def make_fwd(layer):
                def hook(module,inp,out):
                    x=inp[0] if isinstance(inp,tuple) else inp
                    o=out[0] if isinstance(out,tuple) else out
                    self._mlp_in[layer]=x.detach().clone()
                    self._mlp_out[layer]=o.detach().clone()
                return hook
            self._hooks.append(mod.register_forward_hook(make_fwd(li)))
    def _clear_hooks(self):
        for h in self._hooks:h.remove()
        self._hooks=[]
    def _forward_cache(self,input_ids,attention_mask):
        self._setup_hooks()
        self._m.eval()
        with torch.no_grad():
            out=self._m(input_ids=input_ids,attention_mask=attention_mask)
        self._clear_hooks()
        return out
    def _get_active_neurons(self,layer:int,top_k:int)->list[tuple[int,int,float]]:
        if layer not in self._mlp_out:return []
        mlp_o=self._mlp_out[layer]
        down=self._get_down_proj(dict((l,m) for l,_,m in self._mlp_mods)[layer])
        if down is None:return []
        active=[]
        for pos in range(mlp_o.shape[1]):
            out_vec=mlp_o[0,pos,:]
            try:
                contrib=torch.matmul(down,out_vec)
            except RuntimeError:
                try:contrib=torch.matmul(down.T,out_vec)
                except:continue
            topn=torch.topk(contrib.abs(),min(top_k,len(contrib)))
            for ni,act in zip(topn.indices,topn.values):
                active.append((pos,ni.item(),act.item()))
        active.sort(key=lambda x:abs(x[2]),reverse=True)
        return active[:top_k]
    def _compute_attribution_backward(self,input_ids,attention_mask,
                                        target_layer,target_pos,
                                        inject_vec,source_vecs,
                                        source_info)->torch.Tensor:
        self._m.eval()
        for p in self._m.parameters():
            p.requires_grad_(False)
        fwd_hooks=[]
        bwd_hooks=[]
        scores=torch.zeros(len(source_info),device=self._dev)
        grad_cache={}
        residuals={}
        def make_resid_fwd(layer):
            def hook(module,inp,out):
                x=inp[0] if isinstance(inp,tuple) else inp
                o=out[0] if isinstance(out,tuple) else out
                residuals[layer]=o
                if layer==target_layer:
                    o.requires_grad_(True)
                    o.retain_grad()
                    residuals[f"{layer}_grad"]=o
            return hook
        def make_mlp_bwd(layer,src_vecs_layer,src_offset):
            def hook(module,grad_in,grad_out):
                if grad_out is None or grad_out[0] is None:return
                g=grad_out[0]
                for i,(pos,nidx,vec) in enumerate(src_vecs_layer):
                    if pos<g.shape[1]:
                        score=torch.dot(g[0,pos,:].flatten(),vec.flatten())
                        scores[src_offset+i]=score.item()
            return hook
        for li,name,mod in self._mlp_mods:
            fwd_hooks.append(mod.register_forward_hook(make_resid_fwd(li)))
        src_offset=0
        for li,name,mod in self._mlp_mods:
            layer_srcs=[(pos,nidx,vec) for si,(pos,nidx,vec) in enumerate(source_vecs)
                        if source_info[si]["layer"]==li]
            if layer_srcs:
                bwd_hooks.append(mod.register_full_backward_hook(
                    make_mlp_bwd(li,layer_srcs,src_offset)))
                src_offset+=len(layer_srcs)
        try:
            out=self._m(input_ids=input_ids,attention_mask=attention_mask)
            tgt_key=f"{target_layer}_grad"
            if tgt_key in residuals and residuals[tgt_key].grad_fn is not None:
                tgt=residuals[tgt_key]
                grad=torch.zeros_like(tgt)
                grad[0,target_pos,:]=inject_vec.to(grad.device)
                tgt.backward(grad,retain_graph=False)
        except Exception as e:
            log.warning("backward attribution failed: %s",e)
        for h in fwd_hooks+bwd_hooks:h.remove()
        for p in self._m.parameters():
            p.requires_grad_(False)
        return scores
    def trace(self,prompt:str,max_logits:int=5,batch_size:int=32)->AttrGraph:
        enc=self._t(prompt,return_tensors="pt",truncation=True,max_length=256)
        inp={k:v.to(self._dev) for k,v in enc.items() if k in ("input_ids","attention_mask")}
        tokens=[self._t.decode([t]) for t in inp["input_ids"][0]]
        n_pos=len(tokens)
        out=self._forward_cache(inp["input_ids"],inp["attention_mask"])
        logits=out.logits[0,-1,:]
        probs=F.softmax(logits,dim=-1)
        topk=torch.topk(probs,max_logits)
        logit_targets=[{"token":self._t.decode([idx.item()]),"prob":p.item(),
                        "idx":idx.item()} for idx,p in zip(topk.indices,topk.values)]
        all_nodes=[]
        all_decoder_vecs=[]
        all_encoder_vecs=[]
        per_layer_k=max(1,self._k//self._n_layers)
        for li,name,mod in self._mlp_mods:
            active=self._get_active_neurons(li,per_layer_k)
            down=self._get_down_proj(mod)
            gate=self._get_gate_proj(mod)
            for pos,nidx,act in active:
                node={"layer":li,"position":pos,"neuron":nidx,
                      "activation":act,"type":"mlp"}
                all_nodes.append(node)
                if down is not None:
                    try:
                        dec_vec=down[nidx,:] if nidx<down.shape[0] else down[:,nidx]
                    except:
                        dec_vec=torch.zeros(down.shape[-1],device=self._dev)
                    all_decoder_vecs.append((pos,nidx,dec_vec*act))
                if gate is not None:
                    try:
                        enc_vec=gate[nidx,:] if nidx<gate.shape[0] else gate[:,nidx]
                    except:
                        enc_vec=torch.zeros(gate.shape[-1],device=self._dev)
                    all_encoder_vecs.append(enc_vec)
        for i,tok_str in enumerate(tokens):
            all_nodes.append({"layer":-1,"position":i,"neuron":-1,
                              "activation":0,"type":"token","token":tok_str})
        for lt in logit_targets:
            all_nodes.append({"layer":self._n_layers,"position":n_pos-1,
                              "neuron":lt["idx"],"activation":lt["prob"],
                              "type":"logit","token":lt["token"]})
        n_nodes=len(all_nodes)
        adj=torch.zeros(n_nodes,n_nodes)
        mlp_indices=[i for i,nd in enumerate(all_nodes) if nd["type"]=="mlp"]
        tok_indices=[i for i,nd in enumerate(all_nodes) if nd["type"]=="token"]
        log_indices=[i for i,nd in enumerate(all_nodes) if nd["type"]=="logit"]
        lm_head=None
        for n,m in self._m.named_modules():
            if "lm_head" in n:lm_head=m;break
        if lm_head is not None and log_indices and mlp_indices:
            last_layer=max(all_nodes[i]["layer"] for i in mlp_indices)
            last_layer_nodes=[i for i in mlp_indices if all_nodes[i]["layer"]==last_layer]
            for li_idx in log_indices:
                tok_idx=all_nodes[li_idx]["neuron"]
                try:
                    unembed=lm_head.weight[tok_idx,:] if hasattr(lm_head,"weight") else None
                except:unembed=None
                if unembed is None:continue
                for si in last_layer_nodes:
                    if si<len(all_decoder_vecs):
                        _,_,dvec=all_decoder_vecs[si]
                        score=torch.dot(unembed.to(dvec.device),dvec.flatten()[:unembed.shape[0]])
                        adj[li_idx,si]=score.item()
        for i in mlp_indices:
            ni=all_nodes[i]
            for j in mlp_indices:
                if i==j:continue
                nj=all_nodes[j]
                if nj["layer"]>=ni["layer"]:continue
                if nj["position"]>ni["position"]:continue
                if i<len(all_encoder_vecs) and j<len(all_decoder_vecs):
                    evec=all_encoder_vecs[i]
                    _,_,dvec=all_decoder_vecs[j]
                    mn=min(evec.shape[0],dvec.flatten().shape[0])
                    score=torch.dot(evec[:mn].to(dvec.device),dvec.flatten()[:mn])
                    adj[i,j]=score.item()
        for i in mlp_indices:
            ni=all_nodes[i]
            if ni["layer"]==self._layers[0]:
                for j in tok_indices:
                    if all_nodes[j]["position"]<=ni["position"]:
                        adj[i,j]=abs(ni["activation"])*0.01
        influence=self._compute_influence(adj,log_indices)
        g=AttrGraph(prompt=prompt,tokens=tokens,n_layers=self._n_layers,
                     n_pos=n_pos,node_info=all_nodes,adjacency=adj,
                     logit_targets=logit_targets,node_influence=influence)
        return g
    def _compute_influence(self,adj:torch.Tensor,logit_indices:list[int],
                            n_hops:int=3)->torch.Tensor:
        n=adj.shape[0]
        norm=adj.abs()
        row_sums=norm.sum(dim=1,keepdim=True).clamp(min=1e-8)
        norm=norm/row_sums
        logit_weight=torch.zeros(n)
        for li in logit_indices:
            logit_weight[li]=1.0/max(len(logit_indices),1)
        influence=torch.zeros(n)
        power=norm.clone()
        for hop in range(n_hops):
            contrib=logit_weight@power
            influence+=contrib
            power=power@norm
        return influence
    def plot(self,graph:AttrGraph,path:str,title:str=""):
        try:
            import matplotlib;matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except:return
        mlp_nodes=[(i,n) for i,n in enumerate(graph.node_info) if n["type"]=="mlp"]
        if not mlp_nodes:return
        fig,axes=plt.subplots(2,2,figsize=(16,12))
        layers=[n["layer"] for _,n in mlp_nodes]
        acts=[abs(n["activation"]) for _,n in mlp_nodes]
        axes[0,0].scatter(layers,acts,c=layers,cmap="viridis",alpha=0.6,s=20)
        axes[0,0].set_xlabel("Layer");axes[0,0].set_ylabel("|Activation|")
        axes[0,0].set_title("Neuron Activations by Layer")
        if graph.node_influence is not None:
            mlp_inf=[(i,graph.node_influence[i].item()) for i,_ in mlp_nodes]
            mlp_inf.sort(key=lambda x:x[1],reverse=True)
            top20=mlp_inf[:20]
            labels=[f"L{graph.node_info[i]['layer']}N{graph.node_info[i]['neuron']}" for i,_ in top20]
            vals=[v for _,v in top20]
            axes[0,1].barh(range(len(top20)),vals,color="#E91E63")
            axes[0,1].set_yticks(range(len(top20)))
            axes[0,1].set_yticklabels(labels,fontsize=7)
            axes[0,1].set_xlabel("Causal Influence on Output")
            axes[0,1].set_title("Top 20 Most Influential Neurons")
            axes[0,1].invert_yaxis()
        if graph.adjacency is not None:
            adj=graph.adjacency.numpy()
            n=min(adj.shape[0],60)
            im=axes[1,0].imshow(adj[:n,:n],cmap="RdBu_r",aspect="auto")
            axes[1,0].set_title(f"Causal Adjacency (top {n} nodes)")
            plt.colorbar(im,ax=axes[1,0])
        if graph.logit_targets:
            toks=[lt["token"] for lt in graph.logit_targets]
            probs=[lt["prob"] for lt in graph.logit_targets]
            axes[1,1].barh(range(len(toks)),probs,color="#2196F3")
            axes[1,1].set_yticks(range(len(toks)));axes[1,1].set_yticklabels(toks)
            axes[1,1].set_xlabel("Probability")
            axes[1,1].set_title("Output Predictions")
        fig.suptitle(title or f"Circuit Attribution: {graph.prompt[:60]}",fontsize=13)
        plt.tight_layout()
        plt.savefig(path,dpi=150)
        plt.close()
    def compare_and_plot(self,prompt:str,base_model,path:str):
        g_trained=self.trace(prompt)
        base_tracer=ModelCircuitTracer(base_model,self._t,self._k,
                                        next(base_model.parameters()).device)
        g_base=base_tracer.trace(prompt)
        try:
            import matplotlib;matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except:return g_trained,g_base
        fig,axes=plt.subplots(2,2,figsize=(16,12))
        if g_trained.node_influence is not None and g_base.node_influence is not None:
            t_inf={};b_inf={}
            for i,n in enumerate(g_trained.node_info):
                if n["type"]=="mlp":
                    k=(n["layer"],n["neuron"])
                    t_inf[k]=g_trained.node_influence[i].item()
            for i,n in enumerate(g_base.node_info):
                if n["type"]=="mlp":
                    k=(n["layer"],n["neuron"])
                    b_inf[k]=g_base.node_influence[i].item()
            all_keys=set(t_inf.keys())|set(b_inf.keys())
            deltas=[]
            for k in all_keys:
                d=t_inf.get(k,0)-b_inf.get(k,0)
                if abs(d)>0.001:
                    deltas.append({"layer":k[0],"neuron":k[1],"delta":d,
                                   "trained":t_inf.get(k,0),"base":b_inf.get(k,0)})
            deltas.sort(key=lambda x:abs(x["delta"]),reverse=True)
            top=deltas[:20]
            if top:
                colors=["#F44336" if d["delta"]>0 else "#2196F3" for d in top]
                axes[0,0].barh(range(len(top)),[d["delta"] for d in top],color=colors)
                axes[0,0].set_yticks(range(len(top)))
                axes[0,0].set_yticklabels([f"L{d['layer']}N{d['neuron']}" for d in top],fontsize=7)
                axes[0,0].set_xlabel("Influence Delta")
                axes[0,0].set_title("Top Causal Influence Changes")
                axes[0,0].invert_yaxis()
            layer_delta={}
            for d in deltas:
                l=d["layer"]
                layer_delta[l]=layer_delta.get(l,0)+abs(d["delta"])
            if layer_delta:
                ls=sorted(layer_delta.keys())
                axes[0,1].bar(ls,[layer_delta[l] for l in ls],color="#9C27B0",alpha=0.7)
                axes[0,1].set_xlabel("Layer");axes[0,1].set_ylabel("Total |ΔInfluence|")
                axes[0,1].set_title("Causal Change by Layer")
        tl=g_trained.logit_targets;bl=g_base.logit_targets
        if tl and bl:
            all_toks=list({lt["token"] for lt in tl+bl})
            t_probs={lt["token"]:lt["prob"] for lt in tl}
            b_probs={lt["token"]:lt["prob"] for lt in bl}
            x=np.arange(len(all_toks))
            axes[1,0].bar(x-0.15,[b_probs.get(t,0) for t in all_toks],0.3,
                         label="Base",color="#FF9800",alpha=0.7)
            axes[1,0].bar(x+0.15,[t_probs.get(t,0) for t in all_toks],0.3,
                         label="Trained",color="#4CAF50",alpha=0.7)
            axes[1,0].set_xticks(x);axes[1,0].set_xticklabels(all_toks,fontsize=8)
            axes[1,0].legend();axes[1,0].set_title("Logit Shift")
        if g_trained.adjacency is not None and g_base.adjacency is not None:
            mn=min(g_trained.adjacency.shape[0],g_base.adjacency.shape[0],50)
            diff=g_trained.adjacency[:mn,:mn]-g_base.adjacency[:mn,:mn]
            im=axes[1,1].imshow(diff.numpy(),cmap="RdBu_r",aspect="auto")
            axes[1,1].set_title("Adjacency Difference (trained - base)")
            plt.colorbar(im,ax=axes[1,1])
        fig.suptitle(f"Causal Circuit Comparison: {prompt[:50]}",fontsize=13)
        plt.tight_layout()
        plt.savefig(path,dpi=150)
        plt.close()
        return g_trained,g_base
