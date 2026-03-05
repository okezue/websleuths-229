from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def plot_train_loss(hist:list[dict],title:str="",path:str="loss.png"):
    if not hist:return
    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot([h["loss"] for h in hist],linewidth=0.8)
    ax.set_xlabel("Step");ax.set_ylabel("Loss")
    if title:ax.set_title(title)
    fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)

def plot_dream_loss(hist:list[dict],title:str="",path:str="dream.png"):
    if not hist:return
    vals=[h.get("dream_loss",0) for h in hist]
    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot(vals,linewidth=0.8,color="orange")
    ax.set_xlabel("Step");ax.set_ylabel("Dream Loss")
    if title:ax.set_title(title)
    fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)

def plot_lambda_evo(hist:list[dict],title:str="",path:str="lambda.png"):
    if not hist:return
    vals=[h.get("lambda",0) for h in hist]
    if not any(v!=0 for v in vals):return
    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot(vals,linewidth=0.8,color="green")
    ax.set_xlabel("Step");ax.set_ylabel("Lambda")
    if title:ax.set_title(title)
    fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)

def plot_dual_loss(hist:list[dict],title:str="",path:str="dual.png"):
    if not hist:return
    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot([h["loss"] for h in hist],linewidth=0.8,label="Episode")
    ax.plot([h.get("dream_loss",0) for h in hist],linewidth=0.8,label="Dream",color="orange")
    ax.set_xlabel("Step");ax.set_ylabel("Loss")
    ax.legend()
    if title:ax.set_title(title)
    fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)

def plot_mmlu_prog(scores:dict[str,list[float]],title:str="",path:str="mmlu.png"):
    if not scores:return
    fig,ax=plt.subplots(figsize=(10,5))
    for d,vals in scores.items():
        ax.plot(range(len(vals)),vals,marker="o",markersize=3,label=d)
    ax.set_xlabel("Checkpoint");ax.set_ylabel("Accuracy")
    ax.legend();ax.set_ylim(0,1)
    if title:ax.set_title(title)
    fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)

def plot_domain_compare(bl:dict[str,float],fn:dict[str,float],
                        title:str="",path:str="domain_cmp.png"):
    if not bl:return
    doms=list(bl.keys())
    x=np.arange(len(doms));w=0.35
    fig,ax=plt.subplots(figsize=(8,5))
    ax.bar(x-w/2,[bl[d] for d in doms],w,label="Baseline")
    ax.bar(x+w/2,[fn.get(d,0) for d in doms],w,label="Final")
    ax.set_xticks(x);ax.set_xticklabels(doms);ax.legend()
    ax.set_ylabel("Accuracy");ax.set_ylim(0,1)
    if title:ax.set_title(title)
    fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)

def plot_retention_heatmap(mx:dict[str,list[float]],steps:list[str]|None=None,
                           title:str="",path:str="retention.png"):
    if not mx:return
    doms=list(mx.keys())
    lens=[len(mx[d]) for d in doms]
    if not lens or max(lens)==0:return
    ml=max(lens)
    data=np.array([mx[d]+[0.0]*(ml-len(mx[d])) for d in doms])
    fig,ax=plt.subplots(figsize=(max(8,ml*0.6),4))
    im=ax.imshow(data,aspect="auto",cmap="RdYlGn",vmin=0,vmax=max(0.5,data.max()))
    ax.set_yticks(range(len(doms)));ax.set_yticklabels(doms)
    if steps:
        ax.set_xticks(range(len(steps)));ax.set_xticklabels(steps,rotation=45,ha="right",fontsize=7)
    fig.colorbar(im,ax=ax,label="Accuracy")
    if title:ax.set_title(title)
    fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)

def plot_adaptive_steps(topics:list[str],steps:list[int],
                        title:str="",path:str="asteps.png"):
    if not topics:return
    fig,ax=plt.subplots(figsize=(max(8,len(topics)*0.5),4))
    ax.bar(range(len(topics)),steps,color="steelblue")
    ax.set_xticks(range(len(topics)))
    ax.set_xticklabels(topics,rotation=45,ha="right",fontsize=6)
    ax.set_ylabel("Steps")
    if title:ax.set_title(title)
    fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)
