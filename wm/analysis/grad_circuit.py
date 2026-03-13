from __future__ import annotations
import logging,os,json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass,field
from typing import Optional
from contextlib import contextmanager

log=logging.getLogger(__name__)

@dataclass
class GradCircuitGraph:
    prompt:str=""
    tokens:list[str]=field(default_factory=list)
    n_layers:int=0
    n_pos:int=0
    nodes:list[dict]=field(default_factory=list)
    adjacency:Optional[torch.Tensor]=None
    logit_targets:list[dict]=field(default_factory=list)
    influence:Optional[torch.Tensor]=None
    def save(self,path:str):
        torch.save({"prompt":self.prompt,"tokens":self.tokens,
                     "n_layers":self.n_layers,"n_pos":self.n_pos,
                     "nodes":self.nodes,"adjacency":self.adjacency,
                     "logit_targets":self.logit_targets,
                     "influence":self.influence},path)

class GradCircuitTracer:
    def __init__(self,model_name:str,lora_ckpt:str|None=None,
                 tok=None,top_k:int=64,dev:str="cuda:0"):
        self._mn=model_name
        self._ckpt=lora_ckpt
        self._top_k=top_k
        self._dev=torch.device(dev)
        self._tok=tok
        self._model=None
        self._mlp_layers=[]
    def _load_merged(self):
        if self._model is not None:return
        from transformers import AutoModelForCausalLM,AutoTokenizer
        log.info("loading %s in bf16 (full precision for gradient flow)...",self._mn)
        m=AutoModelForCausalLM.from_pretrained(self._mn,torch_dtype=torch.bfloat16,
            trust_remote_code=True).to(self._dev)
        if self._ckpt:
            from peft import PeftModel
            log.info("loading LoRA from %s and merging...",self._ckpt)
            m=PeftModel.from_pretrained(m,self._ckpt)
            m=m.merge_and_unload()
            log.info("LoRA merged into base weights")
        if self._tok is None:
            self._tok=AutoTokenizer.from_pretrained(self._mn,trust_remote_code=True)
            if self._tok.pad_token is None:self._tok.pad_token=self._tok.eos_token
        self._model=m
        self._find_layers()
        log.info("model ready: %d layers, %.1fGB",
                 len(self._mlp_layers),
                 sum(p.numel()*p.element_size() for p in m.parameters())/1e9)
    def _find_layers(self):
        self._mlp_layers=[]
        self._attn_layers=[]
        self._ln_layers=[]
        self._embed=None
        self._lm_head=None
        for name,mod in self._model.named_modules():
            if "lm_head" in name and isinstance(mod,nn.Linear):
                self._lm_head=mod
            if "embed_tokens" in name and isinstance(mod,nn.Embedding):
                self._embed=mod
            parts=name.split(".")
            digits=[p for p in parts if p.isdigit()]
            if not digits:continue
            li=int(digits[0])
            lname=parts[-1] if parts else name
            if lname=="mlp":
                self._mlp_layers.append((li,mod))
            elif lname in ("self_attn","attention"):
                self._attn_layers.append((li,mod))
            elif "norm" in lname or "ln" in lname:
                self._ln_layers.append((li,mod))
    @contextmanager
    def _frozen_context(self):
        orig_grad={}
        for n,p in self._model.named_parameters():
            orig_grad[n]=p.requires_grad
            p.requires_grad_(False)
        try:
            yield
        finally:
            for n,p in self._model.named_parameters():
                p.requires_grad_(orig_grad.get(n,False))
    def _get_mlp_weights(self,mlp)->dict:
        w={}
        for n,p in mlp.named_parameters():
            if "gate" in n:w["gate"]=p
            elif "up" in n:w["up"]=p
            elif "down" in n:w["down"]=p
        if "gate" not in w and "up" not in w:
            params=list(mlp.parameters())
            if len(params)>=2:
                w["up"]=params[0]
                w["down"]=params[-1]
        return w
    def trace(self,prompt:str,max_logits:int=5)->GradCircuitGraph:
        self._load_merged()
        self._model.eval()
        enc=self._tok(prompt,return_tensors="pt",truncation=True,max_length=128)
        input_ids=enc["input_ids"].to(self._dev)
        attn_mask=enc["attention_mask"].to(self._dev)
        tokens=[self._tok.decode([t]) for t in input_ids[0]]
        n_pos=len(tokens)
        mlp_in_cache={};mlp_out_cache={}
        hooks=[]
        for li,mod in self._mlp_layers:
            def make_hook(layer):
                def fwd(module,inp,out):
                    x=inp[0] if isinstance(inp,tuple) else inp
                    o=out[0] if isinstance(out,tuple) else out
                    mlp_in_cache[layer]=x.detach()
                    mlp_out_cache[layer]=o.detach()
                return fwd
            hooks.append(mod.register_forward_hook(make_hook(li)))
        with torch.no_grad():
            out=self._model(input_ids=input_ids,attention_mask=attn_mask)
        for h in hooks:h.remove()
        logits=out.logits[0,-1,:]
        probs=F.softmax(logits.float(),dim=-1)
        topk=torch.topk(probs,max_logits)
        logit_targets=[{"token":self._tok.decode([idx.item()]),"prob":p.item(),
                        "idx":idx.item()} for idx,p in zip(topk.indices,topk.values)]
        nodes=[];encoder_vecs=[];decoder_vecs=[]
        per_layer=max(1,self._top_k//len(self._mlp_layers))
        for li,mod in self._mlp_layers:
            if li not in mlp_out_cache:continue
            w=self._get_mlp_weights(mod)
            down=w.get("down")
            gate=w.get("gate")
            up=w.get("up")
            mlp_o=mlp_out_cache[li]
            for pos in range(n_pos):
                ov=mlp_o[0,pos,:]
                if down is not None:
                    try:
                        if down.data.shape[0]<down.data.shape[1]:
                            contrib=torch.matmul(down.data,ov)
                        else:
                            contrib=torch.matmul(down.data.T,ov)
                    except:continue
                    topn=torch.topk(contrib.abs(),min(per_layer,len(contrib)))
                    for ni,act in zip(topn.indices,topn.values):
                        nidx=ni.item()
                        nodes.append({"layer":li,"position":pos,"neuron":nidx,
                                      "activation":act.item(),"type":"mlp"})
                        if down.data.shape[0]<down.data.shape[1]:
                            dv=down.data[nidx,:]*act
                        else:
                            dv=down.data[:,nidx]*act
                        decoder_vecs.append(dv.detach())
                        enc=gate.data[nidx,:] if gate is not None else (up.data[nidx,:] if up is not None else torch.zeros_like(dv))
                        encoder_vecs.append(enc.detach())
        nodes.sort(key=lambda x:abs(x["activation"]),reverse=True)
        nodes=nodes[:self._top_k]
        encoder_vecs=encoder_vecs[:self._top_k]
        decoder_vecs=decoder_vecs[:self._top_k]
        for i,t in enumerate(tokens):
            nodes.append({"layer":-1,"position":i,"neuron":-1,"activation":0,"type":"token","token":t})
        for lt in logit_targets:
            nodes.append({"layer":len(self._mlp_layers),"position":n_pos-1,
                          "neuron":lt["idx"],"activation":lt["prob"],"type":"logit","token":lt["token"]})
        n_mlp=len([n for n in nodes if n["type"]=="mlp"])
        n_all=len(nodes)
        adj=torch.zeros(n_all,n_all,device=self._dev)
        log.info("computing gradient attribution for %d mlp nodes...",n_mlp)
        with self._frozen_context():
            for tgt_i in range(n_mlp):
                tgt=nodes[tgt_i]
                if tgt_i>=len(encoder_vecs):break
                evec=encoder_vecs[tgt_i]
                tgt_layer=tgt["layer"]
                tgt_pos=tgt["position"]
                resid_hooks=[];bwd_hooks=[];resid_cache={}
                for li,mod in self._mlp_layers:
                    def make_resid(layer):
                        def fwd(module,inp,out):
                            o=out[0] if isinstance(out,tuple) else out
                            if layer==tgt_layer:
                                o=o.detach().requires_grad_(True)
                                resid_cache[layer]=o
                            return o
                        return fwd
                    resid_hooks.append(mod.register_forward_hook(make_resid(li)))
                score_buf=torch.zeros(n_all,device=self._dev)
                for li,mod in self._mlp_layers:
                    if li>=tgt_layer:continue
                    src_indices=[j for j in range(n_mlp) if nodes[j]["layer"]==li and j<len(decoder_vecs)]
                    if not src_indices:continue
                    def make_bwd(layer,indices):
                        def hook(module,grad_in,grad_out):
                            if grad_out is None or grad_out[0] is None:return
                            g=grad_out[0]
                            for j in indices:
                                p=nodes[j]["position"]
                                if p<g.shape[1] and j<len(decoder_vecs):
                                    dv=decoder_vecs[j]
                                    mn=min(g.shape[2],dv.shape[0])
                                    s=torch.dot(g[0,p,:mn],dv[:mn])
                                    score_buf[j]=s.item()
                        return hook
                    bwd_hooks.append(mod.register_full_backward_hook(make_bwd(li,src_indices)))
                try:
                    o2=self._model(input_ids=input_ids,attention_mask=attn_mask)
                    if tgt_layer in resid_cache:
                        r=resid_cache[tgt_layer]
                        if r.grad_fn is not None:
                            grad=torch.zeros_like(r)
                            mn=min(grad.shape[2],evec.shape[0])
                            grad[0,tgt_pos,:mn]=evec[:mn]
                            r.backward(grad,retain_graph=False)
                except Exception as e:
                    log.debug("backward failed for node %d: %s",tgt_i,e)
                for h in resid_hooks+bwd_hooks:h.remove()
                adj[tgt_i,:]=score_buf
                if tgt_i%10==0:
                    log.info("  traced %d/%d nodes",tgt_i,n_mlp)
        logit_indices=[i for i,n in enumerate(nodes) if n["type"]=="logit"]
        if self._lm_head is not None:
            for li in logit_indices:
                tok_idx=nodes[li]["neuron"]
                try:unembed=self._lm_head.weight[tok_idx,:]
                except:continue
                last_mlp=max((nodes[j]["layer"] for j in range(n_mlp)),default=-1)
                for j in range(n_mlp):
                    if nodes[j]["layer"]==last_mlp and j<len(decoder_vecs):
                        dv=decoder_vecs[j]
                        mn=min(unembed.shape[0],dv.shape[0])
                        adj[li,j]=torch.dot(unembed[:mn],dv[:mn]).item()
        influence=self._influence(adj,logit_indices)
        g=GradCircuitGraph(prompt=prompt,tokens=tokens,n_layers=len(self._mlp_layers),
                            n_pos=n_pos,nodes=nodes,adjacency=adj.cpu(),
                            logit_targets=logit_targets,influence=influence.cpu())
        return g
    def _influence(self,adj,logit_idx,hops=3):
        n=adj.shape[0]
        norm=adj.abs()
        rs=norm.sum(1,keepdim=True).clamp(min=1e-8)
        norm=norm/rs
        lw=torch.zeros(n,device=adj.device)
        for i in logit_idx:lw[i]=1.0/max(len(logit_idx),1)
        inf=torch.zeros(n,device=adj.device)
        pw=norm.clone()
        for _ in range(hops):
            inf+=lw@pw
            pw=pw@norm
        return inf
    def plot(self,g:GradCircuitGraph,path:str,title:str=""):
        try:
            import matplotlib;matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except:return
        mlp=[i for i,n in enumerate(g.nodes) if n["type"]=="mlp"]
        if not mlp:return
        fig,axes=plt.subplots(2,2,figsize=(16,12))
        if g.influence is not None:
            top=sorted(mlp,key=lambda i:g.influence[i].item(),reverse=True)[:25]
            labels=[f"L{g.nodes[i]['layer']}P{g.nodes[i]['position']}N{g.nodes[i]['neuron']}" for i in top]
            vals=[g.influence[i].item() for i in top]
            colors=["#E91E63" if v>0 else "#2196F3" for v in vals]
            axes[0,0].barh(range(len(top)),vals,color=colors)
            axes[0,0].set_yticks(range(len(top)));axes[0,0].set_yticklabels(labels,fontsize=6)
            axes[0,0].set_xlabel("Causal Influence");axes[0,0].set_title("Top 25 Influential Neurons")
            axes[0,0].invert_yaxis()
        layer_inf={}
        if g.influence is not None:
            for i in mlp:
                l=g.nodes[i]["layer"]
                layer_inf[l]=layer_inf.get(l,0)+g.influence[i].item()
        if layer_inf:
            ls=sorted(layer_inf.keys())
            axes[0,1].bar(ls,[layer_inf[l] for l in ls],color="#9C27B0",alpha=0.7)
            axes[0,1].set_xlabel("Layer");axes[0,1].set_ylabel("Total Influence")
            axes[0,1].set_title("Causal Influence by Layer")
        if g.adjacency is not None:
            n=min(g.adjacency.shape[0],50)
            im=axes[1,0].imshow(g.adjacency[:n,:n].numpy(),cmap="RdBu_r",aspect="auto")
            axes[1,0].set_title(f"Causal Adjacency Matrix ({n}x{n})")
            plt.colorbar(im,ax=axes[1,0])
        if g.logit_targets:
            toks=[lt["token"] for lt in g.logit_targets]
            probs=[lt["prob"] for lt in g.logit_targets]
            axes[1,1].barh(range(len(toks)),probs,color="#4CAF50")
            axes[1,1].set_yticks(range(len(toks)));axes[1,1].set_yticklabels(toks)
            axes[1,1].set_title("Output Predictions")
        fig.suptitle(title or f"Gradient Circuit: {g.prompt[:60]}",fontsize=13)
        plt.tight_layout()
        plt.savefig(path,dpi=150)
        plt.close()
