from __future__ import annotations
import logging,os,json,math
import torch
import torch.nn.functional as F
from dataclasses import dataclass,field

log=logging.getLogger(__name__)

@dataclass
class LayerStats:
    layer:int=0
    attn_entropy:float=0.0
    mlp_norm:float=0.0
    lora_a_norm:float=0.0
    lora_b_norm:float=0.0
    lora_rank_util:float=0.0
    hidden_norm:float=0.0

def get_attention_maps(model,tok,prompt:str,dev=None):
    if dev is None:dev=next(model.parameters()).device
    model.eval()
    enc=tok(prompt,return_tensors="pt",truncation=True,max_length=256)
    inp={k:v.to(dev) for k,v in enc.items() if k in ("input_ids","attention_mask")}
    with torch.no_grad():
        out=model(**inp,output_attentions=True)
    attns=out.attentions if hasattr(out,"attentions") and out.attentions else []
    maps=[]
    for li,a in enumerate(attns):
        maps.append({"layer":li,"shape":list(a.shape),
                     "mean_entropy":-(a*a.clamp(min=1e-12).log()).sum(-1).mean().item(),
                     "max_attn":a.max().item()})
    return maps

def get_hidden_states(model,tok,prompt:str,dev=None):
    if dev is None:dev=next(model.parameters()).device
    model.eval()
    enc=tok(prompt,return_tensors="pt",truncation=True,max_length=256)
    inp={k:v.to(dev) for k,v in enc.items() if k in ("input_ids","attention_mask")}
    with torch.no_grad():
        out=model(**inp,output_hidden_states=True)
    states=out.hidden_states if hasattr(out,"hidden_states") and out.hidden_states else []
    norms=[]
    for li,h in enumerate(states):
        norms.append({"layer":li,"norm":h.norm().item(),"mean":h.mean().item(),
                      "std":h.std().item(),"shape":list(h.shape)})
    return norms

def logit_lens(model,tok,prompt:str,top_k:int=5,dev=None):
    if dev is None:dev=next(model.parameters()).device
    model.eval()
    enc=tok(prompt,return_tensors="pt",truncation=True,max_length=256)
    inp={k:v.to(dev) for k,v in enc.items() if k in ("input_ids","attention_mask")}
    with torch.no_grad():
        out=model(**inp,output_hidden_states=True)
    states=out.hidden_states if hasattr(out,"hidden_states") and out.hidden_states else []
    lm_head=None
    for n,m in model.named_modules():
        if "lm_head" in n:lm_head=m;break
    if lm_head is None:
        return []
    results=[]
    for li,h in enumerate(states):
        with torch.no_grad():
            logits=lm_head(h[0,-1:,:])
            probs=F.softmax(logits,dim=-1)
            topk=torch.topk(probs[0],top_k)
            tokens=[tok.decode([idx.item()]) for idx in topk.indices]
            results.append({"layer":li,
                           "top_tokens":tokens,
                           "top_probs":[p.item() for p in topk.values]})
    return results

def analyze_lora_weights(model)->list[dict]:
    layers=[]
    for n,p in model.named_parameters():
        if "lora_A" in n:
            bname=n.replace("lora_A","lora_B")
            bp=None
            for n2,p2 in model.named_parameters():
                if n2==bname:bp=p2;break
            a_norm=p.data.norm().item()
            b_norm=bp.data.norm().item() if bp is not None else 0
            dw=bp.data@p.data if bp is not None else None
            svd_spec=[]
            rank_util=0.0
            if dw is not None:
                try:
                    s=torch.linalg.svdvals(dw.float())
                    svd_spec=[v.item() for v in s[:10]]
                    total=s.sum().item()
                    if total>0:
                        cumsum=torch.cumsum(s,0)
                        effective=((cumsum/total)<0.99).sum().item()+1
                        rank_util=effective/len(s)
                except:pass
            base=n.split(".lora_A")[0]
            layers.append({"name":base,"a_norm":a_norm,"b_norm":b_norm,
                           "dw_norm":dw.norm().item() if dw is not None else 0,
                           "svd_top10":svd_spec,"rank_util":rank_util,
                           "r":p.shape[0]})
    return layers

def activation_diff(model,base_model,tok,prompt:str,dev=None):
    if dev is None:dev=next(model.parameters()).device
    model.eval();base_model.eval()
    enc=tok(prompt,return_tensors="pt",truncation=True,max_length=256)
    inp={k:v.to(dev) for k,v in enc.items() if k in ("input_ids","attention_mask")}
    with torch.no_grad():
        out1=model(**inp,output_hidden_states=True)
        b_dev=next(base_model.parameters()).device
        inp2={k:v.to(b_dev) for k,v in enc.items() if k in ("input_ids","attention_mask")}
        out2=base_model(**inp2,output_hidden_states=True)
    s1=out1.hidden_states if out1.hidden_states else []
    s2=out2.hidden_states if out2.hidden_states else []
    diffs=[]
    for li,(h1,h2) in enumerate(zip(s1,s2)):
        h2d=h2.to(dev)
        cos=F.cosine_similarity(h1.flatten(),h2d.flatten(),dim=0).item()
        l2=(h1-h2d).norm().item()
        diffs.append({"layer":li,"cosine_sim":cos,"l2_diff":l2,
                      "relative_diff":l2/max(h1.norm().item(),1e-8)})
    return diffs

def full_inspection(model,base_model,tok,prompts:dict[str,list[str]],
                     out_dir:str,dev=None):
    os.makedirs(out_dir,exist_ok=True)
    report={}
    log.info("analyzing LoRA weights...")
    lora=analyze_lora_weights(model)
    report["lora_weights"]=lora
    report["probes"]={}
    for domain,ps in prompts.items():
        log.info("inspecting %s (%d probes)...",domain,len(ps))
        dom_data=[]
        for p in ps:
            entry={"prompt":p}
            try:entry["attention"]=get_attention_maps(model,tok,p,dev)
            except Exception as e:entry["attention_error"]=str(e)
            try:entry["hidden_states"]=get_hidden_states(model,tok,p,dev)
            except Exception as e:entry["hidden_error"]=str(e)
            try:entry["logit_lens"]=logit_lens(model,tok,p,dev=dev)
            except Exception as e:entry["logit_lens_error"]=str(e)
            if base_model:
                try:entry["activation_diff"]=activation_diff(model,base_model,tok,p,dev)
                except Exception as e:entry["diff_error"]=str(e)
            dom_data.append(entry)
        report["probes"][domain]=dom_data
    with open(f"{out_dir}/deep_inspect.json","w") as f:
        json.dump(report,f,indent=2,default=str)
    _plot_inspection(report,out_dir)
    return report

def _plot_inspection(report,out_dir):
    try:
        import matplotlib;matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except:return
    lora=report.get("lora_weights",[])
    if lora:
        fig,axes=plt.subplots(2,2,figsize=(16,10))
        names=[l["name"].split(".")[-2]+"_"+l["name"].split(".")[-1] if "." in l["name"] else l["name"] for l in lora]
        short=[n[-20:] for n in names]
        axes[0,0].bar(range(len(lora)),[l["dw_norm"] for l in lora],color="#2196F3",alpha=0.7)
        axes[0,0].set_title("LoRA ΔW Norm per Layer");axes[0,0].set_ylabel("||B·A||")
        axes[0,0].set_xticks(range(0,len(lora),max(1,len(lora)//10)))
        axes[0,1].bar(range(len(lora)),[l["rank_util"] for l in lora],color="#4CAF50",alpha=0.7)
        axes[0,1].set_title("Effective Rank Utilization");axes[0,1].set_ylabel("ratio")
        for i,l in enumerate(lora[:4]):
            if l.get("svd_top10"):
                axes[1,0].plot(l["svd_top10"],label=short[i][:15],linewidth=2)
        axes[1,0].set_title("SVD Spectrum (top 10 singular values)");axes[1,0].legend(fontsize=7)
        axes[1,0].set_xlabel("Component");axes[1,0].set_ylabel("σ")
        a_norms=[l["a_norm"] for l in lora]
        b_norms=[l["b_norm"] for l in lora]
        x=np.arange(len(lora))
        axes[1,1].bar(x-0.15,a_norms,0.3,label="A",color="#FF9800",alpha=0.7)
        axes[1,1].bar(x+0.15,b_norms,0.3,label="B",color="#E91E63",alpha=0.7)
        axes[1,1].set_title("LoRA A vs B Norms");axes[1,1].legend()
        plt.tight_layout()
        plt.savefig(f"{out_dir}/lora_analysis.png",dpi=150)
        plt.close()
    probes=report.get("probes",{})
    for domain,entries in probes.items():
        for entry in entries:
            diffs=entry.get("activation_diff",[])
            if diffs:
                fig,axes=plt.subplots(1,2,figsize=(14,5))
                layers=[d["layer"] for d in diffs]
                cos=[d["cosine_sim"] for d in diffs]
                rel=[d["relative_diff"] for d in diffs]
                axes[0].plot(layers,cos,color="#2196F3",linewidth=2)
                axes[0].fill_between(layers,cos,1.0,alpha=0.1,color="#2196F3")
                axes[0].set_title(f"Cosine Sim (base vs finetuned)\n{entry['prompt'][:50]}")
                axes[0].set_xlabel("Layer");axes[0].set_ylabel("Cosine Similarity")
                axes[0].set_ylim(0.5,1.05)
                axes[1].plot(layers,rel,color="#F44336",linewidth=2)
                axes[1].fill_between(layers,rel,alpha=0.1,color="#F44336")
                axes[1].set_title("Relative Activation Difference")
                axes[1].set_xlabel("Layer");axes[1].set_ylabel("||Δh|| / ||h||")
                plt.tight_layout()
                slug=entry["prompt"][:20].replace(" ","_").replace("/","")
                plt.savefig(f"{out_dir}/actdiff_{domain}_{slug}.png",dpi=150)
                plt.close()
                break
            ll=entry.get("logit_lens",[])
            if ll:
                fig,ax=plt.subplots(figsize=(14,6))
                layers=[l["layer"] for l in ll]
                top1=[l["top_tokens"][0] if l["top_tokens"] else "" for l in ll]
                top1_prob=[l["top_probs"][0] if l["top_probs"] else 0 for l in ll]
                ax.bar(layers,top1_prob,color="#9C27B0",alpha=0.7)
                for i,t in enumerate(top1):
                    if i%3==0:
                        ax.annotate(t,xy=(layers[i],top1_prob[i]),fontsize=6,rotation=45,ha="left")
                ax.set_title(f"Logit Lens: Top-1 Token Probability per Layer\n{entry['prompt'][:50]}")
                ax.set_xlabel("Layer");ax.set_ylabel("P(top token)")
                plt.tight_layout()
                slug=entry["prompt"][:20].replace(" ","_").replace("/","")
                plt.savefig(f"{out_dir}/logitlens_{domain}_{slug}.png",dpi=150)
                plt.close()
                break
