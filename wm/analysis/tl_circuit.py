from __future__ import annotations
import logging,os,json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass,field
from typing import Optional

log=logging.getLogger(__name__)

@dataclass
class TLCircuitGraph:
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
    @classmethod
    def load(cls,path:str)->"TLCircuitGraph":
        d=torch.load(path,map_location="cpu",weights_only=False)
        g=cls(prompt=d["prompt"],tokens=d["tokens"],n_layers=d["n_layers"],
              n_pos=d["n_pos"],nodes=d["nodes"],logit_targets=d["logit_targets"])
        g.adjacency=d.get("adjacency")
        g.influence=d.get("influence")
        return g

def _stop_grad(x,hook):
    return x.detach().requires_grad_(x.requires_grad)

class TLCircuitTracer:
    def __init__(self,model_name:str,lora_ckpt:str|None=None,
                 top_k:int=100,dev:str="cuda:0",dtype=torch.bfloat16):
        self._mn=model_name
        self._ckpt=lora_ckpt
        self._k=top_k
        self._dev=dev
        self._dtype=dtype
        self._tl=None
        self._tok=None
    def _load(self):
        if self._tl is not None:return
        import transformer_lens as tl
        from transformer_lens import HookedTransformer
        if self._ckpt:
            from transformers import AutoModelForCausalLM
            from peft import PeftModel
            log.info("loading %s + merging LoRA from %s...",self._mn,self._ckpt)
            hf=AutoModelForCausalLM.from_pretrained(self._mn,torch_dtype=self._dtype,
                                                     trust_remote_code=True)
            hf=PeftModel.from_pretrained(hf,self._ckpt)
            hf=hf.merge_and_unload()
            self._tl=HookedTransformer.from_pretrained(self._mn,hf_model=hf,
                                                        device=self._dev,dtype=self._dtype)
            del hf;torch.cuda.empty_cache()
        else:
            log.info("loading %s into TransformerLens...",self._mn)
            self._tl=HookedTransformer.from_pretrained(self._mn,device=self._dev,
                                                        dtype=self._dtype)
        self._tok=self._tl.tokenizer
        log.info("TL model loaded: %d layers, d_model=%d, d_mlp=%d",
                 self._tl.cfg.n_layers,self._tl.cfg.d_model,
                 self._tl.cfg.d_mlp if hasattr(self._tl.cfg,"d_mlp") else 0)
    def _get_active_neurons(self,mlp_post,layer:int,top_k:int)->list[tuple[int,int,float]]:
        active=[]
        for pos in range(mlp_post.shape[1]):
            acts=mlp_post[0,pos,:]
            topn=torch.topk(acts.abs(),min(top_k,len(acts)))
            for ni,a in zip(topn.indices,topn.values):
                active.append((pos,ni.item(),a.item()))
        active.sort(key=lambda x:abs(x[2]),reverse=True)
        return active[:top_k]
    def trace(self,prompt:str,max_logits:int=5,batch_size:int=64)->TLCircuitGraph:
        self._load()
        tl=self._tl
        dev=torch.device(self._dev)
        tokens=tl.to_tokens(prompt)
        str_tokens=tl.to_str_tokens(prompt)
        n_pos=tokens.shape[1]
        n_layers=tl.cfg.n_layers
        d_model=tl.cfg.d_model
        d_mlp=tl.cfg.d_mlp if hasattr(tl.cfg,"d_mlp") else tl.cfg.d_model*4
        log.info("tracing '%s' (%d tokens, %d layers)...",prompt[:40],n_pos,n_layers)
        cache={}
        def cache_hook(name):
            def hook_fn(act,hook):
                cache[name]=act.detach()
            return hook_fn
        fwd_hooks=[]
        for li in range(n_layers):
            fwd_hooks.append((f"blocks.{li}.hook_mlp_out",cache_hook(f"mlp_out_{li}")))
            fwd_hooks.append((f"blocks.{li}.mlp.hook_post",cache_hook(f"mlp_post_{li}")))
            fwd_hooks.append((f"blocks.{li}.hook_resid_post",cache_hook(f"resid_{li}")))
        with torch.no_grad():
            logits=tl.run_with_hooks(tokens,fwd_hooks=fwd_hooks)
        probs=F.softmax(logits[0,-1,:].float(),dim=-1)
        topk=torch.topk(probs,max_logits)
        logit_targets=[]
        for idx,p in zip(topk.indices,topk.values):
            tok_str=tl.to_string(idx.unsqueeze(0)) if hasattr(tl,"to_string") else str(idx.item())
            logit_targets.append({"token":tok_str,"prob":p.item(),"idx":idx.item()})
        all_nodes=[]
        node_enc=[]
        node_dec=[]
        per_layer=max(1,self._k//n_layers)
        W_out=tl.W_out if hasattr(tl,"W_out") else None
        W_in=tl.W_in if hasattr(tl,"W_in") else None
        for li in range(n_layers):
            post_key=f"mlp_post_{li}"
            if post_key not in cache:continue
            mlp_post=cache[post_key]
            active=self._get_active_neurons(mlp_post,li,per_layer)
            for pos,nidx,act in active:
                all_nodes.append({"layer":li,"position":pos,"neuron":nidx,
                                  "activation":act,"type":"mlp"})
                dec=torch.zeros(d_model,device=dev)
                if W_out is not None and nidx<W_out.shape[1]:
                    dec=W_out[li,nidx,:]*act
                else:
                    for n,p in tl.blocks[li].mlp.named_parameters():
                        if "W_out" in n or "down" in n:
                            if nidx<p.data.shape[0]:
                                dec=p.data[nidx,:d_model]*act
                            elif nidx<p.data.shape[1]:
                                dec=p.data[:d_model,nidx]*act
                            break
                node_dec.append(dec)
                enc=torch.zeros(d_model,device=dev)
                if W_in is not None and nidx<W_in.shape[1]:
                    enc=W_in[li,nidx,:]
                else:
                    for n,p in tl.blocks[li].mlp.named_parameters():
                        if "W_in" in n or "gate" in n or "up" in n:
                            if nidx<p.data.shape[0]:
                                enc=p.data[nidx,:d_model]
                            elif nidx<p.data.shape[1]:
                                enc=p.data[:d_model,nidx]
                            break
                node_enc.append(enc)
        all_nodes.sort(key=lambda x:abs(x["activation"]),reverse=True)
        all_nodes=all_nodes[:self._k]
        node_enc=node_enc[:self._k]
        node_dec=node_dec[:self._k]
        n_mlp=len(all_nodes)
        for i,t in enumerate(str_tokens):
            all_nodes.append({"layer":-1,"position":i,"neuron":-1,"activation":0,"type":"token","token":t})
        for lt in logit_targets:
            all_nodes.append({"layer":n_layers,"position":n_pos-1,
                              "neuron":lt["idx"],"activation":lt["prob"],
                              "type":"logit","token":lt["token"]})
        n_all=len(all_nodes)
        adj=torch.zeros(n_all,n_all,device=dev)
        log.info("backward attribution for %d target nodes (batch=%d)...",n_mlp,batch_size)
        for tgt_batch_start in range(0,n_mlp,batch_size):
            tgt_batch_end=min(tgt_batch_start+batch_size,n_mlp)
            bs=tgt_batch_end-tgt_batch_start
            batch_tokens=tokens.expand(bs,-1)
            score_buf=torch.zeros(bs,n_all,device=dev)
            def make_resid_freeze(layer):
                def hook_fn(act,hook):
                    return act.detach()
                return hook_fn
            def make_attn_freeze(layer):
                def hook_fn(act,hook):
                    return act.detach()
                return hook_fn
            def make_embed_grad(act,hook):
                act.requires_grad_(True)
                return act
            def make_inject(tgt_start,tgt_end,encs,nodes_info):
                def hook_fn(act,hook):
                    for bi in range(tgt_end-tgt_start):
                        li=nodes_info[tgt_start+bi]["layer"]
                        if f"blocks.{li}.hook_mlp_out"==hook.name:
                            pos=nodes_info[tgt_start+bi]["position"]
                            mn=min(act.shape[2],encs[tgt_start+bi].shape[0])
                            grad=torch.zeros_like(act[bi:bi+1])
                            grad[0,pos,:mn]=encs[tgt_start+bi][:mn]
                            act[bi:bi+1]=act[bi:bi+1]+grad*0
                    return act
                return hook_fn
            def make_score_hook(layer,src_decs,src_nodes,tgt_start,tgt_end):
                def hook_fn(grad,hook):
                    if grad is None:return
                    for bi in range(tgt_end-tgt_start):
                        for si in range(len(src_nodes)):
                            if src_nodes[si]["layer"]==layer:
                                pos=src_nodes[si]["position"]
                                if pos<grad.shape[1] and si<len(src_decs):
                                    mn=min(grad.shape[2],src_decs[si].shape[0])
                                    s=torch.dot(grad[bi,pos,:mn],src_decs[si][:mn])
                                    score_buf[bi,si]+=s.item()
                return hook_fn
            run_hooks=[]
            for li in range(n_layers):
                run_hooks.append((f"blocks.{li}.attn.hook_pattern",make_attn_freeze(li)))
                if hasattr(tl.blocks[li],"ln1"):
                    run_hooks.append((f"blocks.{li}.ln1.hook_scale",make_resid_freeze(li)))
                if hasattr(tl.blocks[li],"ln2"):
                    run_hooks.append((f"blocks.{li}.ln2.hook_scale",make_resid_freeze(li)))
            run_hooks.append(("hook_embed",make_embed_grad))
            bwd_hooks=[]
            for li in range(n_layers):
                bwd_hooks.append((f"blocks.{li}.hook_mlp_out",
                    make_score_hook(li,node_dec,all_nodes[:n_mlp],
                                    tgt_batch_start,tgt_batch_end)))
            try:
                logits_out=tl.run_with_hooks(batch_tokens,
                    fwd_hooks=run_hooks,bwd_hooks=bwd_hooks,
                    return_type="logits")
                for bi in range(bs):
                    tgt_i=tgt_batch_start+bi
                    tgt_node=all_nodes[tgt_i]
                    layer=tgt_node["layer"]
                    pos=tgt_node["position"]
                    target=torch.zeros_like(logits_out[bi:bi+1])
                    enc=node_enc[tgt_i]
                    resid_key=f"blocks.{layer}.hook_resid_post"
                    loss=logits_out[bi,-1,logit_targets[0]["idx"]]
                    loss.backward(retain_graph=(bi<bs-1))
            except Exception as e:
                log.warning("backward batch %d failed: %s",tgt_batch_start,e)
            for bi in range(bs):
                adj[tgt_batch_start+bi,:]=score_buf[bi]
            if tgt_batch_start%32==0:
                log.info("  attributed %d/%d",tgt_batch_start,n_mlp)
        logit_idx=[i for i,n in enumerate(all_nodes) if n["type"]=="logit"]
        W_U=tl.W_U if hasattr(tl,"W_U") else None
        if W_U is not None:
            for li in logit_idx:
                tok_idx=all_nodes[li]["neuron"]
                unembed=W_U[:,tok_idx]
                last_layer=max((all_nodes[j]["layer"] for j in range(n_mlp)),default=-1)
                for j in range(n_mlp):
                    if all_nodes[j]["layer"]==last_layer and j<len(node_dec):
                        mn=min(unembed.shape[0],node_dec[j].shape[0])
                        adj[li,j]=torch.dot(unembed[:mn],node_dec[j][:mn]).item()
        influence=self._influence(adj,logit_idx)
        g=TLCircuitGraph(prompt=prompt,tokens=list(str_tokens),
                          n_layers=n_layers,n_pos=n_pos,
                          nodes=all_nodes,adjacency=adj.cpu(),
                          logit_targets=logit_targets,influence=influence.cpu())
        log.info("traced: %d nodes, adj %s, influence range [%.4f, %.4f]",
                 len(all_nodes),list(adj.shape),
                 influence.min().item(),influence.max().item())
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
    def plot(self,g:TLCircuitGraph,path:str,title:str=""):
        try:
            import matplotlib;matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except:return
        mlp=[i for i,n in enumerate(g.nodes) if n["type"]=="mlp"]
        fig,axes=plt.subplots(2,2,figsize=(16,12))
        if g.influence is not None and mlp:
            top=sorted(mlp,key=lambda i:g.influence[i].item(),reverse=True)[:25]
            labels=[f"L{g.nodes[i]['layer']}P{g.nodes[i]['position']}N{g.nodes[i]['neuron']}" for i in top]
            vals=[g.influence[i].item() for i in top]
            axes[0,0].barh(range(len(top)),vals,color="#E91E63")
            axes[0,0].set_yticks(range(len(top)));axes[0,0].set_yticklabels(labels,fontsize=6)
            axes[0,0].set_xlabel("Causal Influence")
            axes[0,0].set_title("Top 25 Most Influential Neurons")
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
            axes[0,1].set_title("Causal Influence Distribution Across Layers")
        if g.adjacency is not None:
            n=min(g.adjacency.shape[0],60)
            a=g.adjacency[:n,:n]
            vmax=max(a.abs().max().item(),1e-8)
            im=axes[1,0].imshow(a.numpy(),cmap="RdBu_r",aspect="auto",vmin=-vmax,vmax=vmax)
            axes[1,0].set_title(f"Causal Adjacency ({n}x{n})")
            plt.colorbar(im,ax=axes[1,0])
        if g.logit_targets:
            toks=[lt["token"] for lt in g.logit_targets]
            probs=[lt["prob"] for lt in g.logit_targets]
            axes[1,1].barh(range(len(toks)),probs,color="#4CAF50")
            axes[1,1].set_yticks(range(len(toks)));axes[1,1].set_yticklabels(toks)
            axes[1,1].set_title("Predicted Next Tokens")
        fig.suptitle(title or f"TL Circuit: {g.prompt[:60]}",fontsize=13)
        plt.tight_layout()
        plt.savefig(path,dpi=150)
        plt.close()
