"""Debug script to diagnose train_loss=0.0 in ablation results.

Run on Colab (same environment as the original ablation):
    !python websleuths-229/scripts/debug_ablation.py --model meta-llama/Llama-3.2-1B

What this checks:
    1. Dataset columns and content after DatasetBuilder
    2. Batch shapes/values coming out of make_ep_dl / WeightedLMCollator
    3. weighted_ce internals: mask, per-sample loss, final value
    4. Whether train_loss is actually zero or just looks zero

Runs only 3 training steps on the naive config (no dreaming).
Takes ~2 minutes on Colab T4.
"""
from __future__ import annotations
import os,sys,copy,argparse
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

sys.path.insert(0,os.path.join(os.path.dirname(__file__),".."))

from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import get_peft_model,LoraConfig,TaskType

from wm.types import Episode
from wm.store import EpisodeStore
from wm.chunk import Chunker
from wm.dataset import DatasetBuilder
from wm.recipe.base import WeightedLMCollator,make_ep_dl,weighted_ce
from wm.cfg import ChunkCfg,DatasetCfg

MODEL_NAME=None
TEXTS=[
    "Forensic DNA analysis has revolutionized criminal investigations. Modern techniques like touch DNA and familial searching allow investigators to identify suspects from trace amounts of biological material left at crime scenes.",
    "Cold case investigations have seen renewed interest with advances in genetic genealogy. Investigators can now use public DNA databases to identify suspects in decades-old unsolved cases through distant relative matching.",
    "The development of rapid DNA technology enables crime labs to process DNA samples in under two hours, compared to the traditional timeline of weeks or months.",
    "Forensic science continues to evolve with new methods for analyzing digital evidence, including smartphone forensics, cloud data extraction, and AI-assisted pattern recognition.",
    "Recent advances in forensic toxicology include more sensitive detection methods for novel psychoactive substances and improved techniques for post-mortem drug analysis.",
]

def sep(msg):
    print(f"\n{'='*60}\n{msg}\n{'='*60}")

def make_episodes():
    return [Episode(url=f"https://example.com/{i}",title=f"Article {i}",
                    body=t,authority=0.8) for i,t in enumerate(TEXTS)]

def make_model(mn):
    tok=AutoTokenizer.from_pretrained(mn,trust_remote_code=True)
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model=AutoModelForCausalLM.from_pretrained(mn,torch_dtype=dtype,trust_remote_code=True)
    if torch.cuda.is_available():model=model.cuda()
    lc=LoraConfig(r=8,lora_alpha=16,lora_dropout=0.05,
                  target_modules=["q_proj","v_proj"],task_type=TaskType.CAUSAL_LM)
    model=get_peft_model(model,lc)
    return model,tok

def check_1_dataset(chunks):
    sep("CHECK 1: Dataset content")
    ds=DatasetBuilder(DatasetCfg(recipe="cpt")).build(chunks)
    print(f"  rows: {len(ds)}")
    print(f"  columns: {ds.column_names}")
    for i in range(min(2,len(ds))):
        row=ds[i]
        print(f"  row[{i}] text[:80]: {repr(row['text'][:80])}")
        print(f"  row[{i}] authority: {row['authority']}")
    return ds

def check_2_dataloader(ds,tok):
    sep("CHECK 2: DataLoader batch internals")
    col=WeightedLMCollator(tok)
    dl=make_ep_dl(ds,tok,col,bs=2,max_len=128)
    batch=next(iter(dl))
    print(f"  batch keys: {list(batch.keys())}")
    print(f"  input_ids shape:     {batch['input_ids'].shape}  dtype: {batch['input_ids'].dtype}")
    print(f"  attention_mask shape:{batch['attention_mask'].shape}  dtype: {batch['attention_mask'].dtype}")
    print(f"  weights:             {batch['weights']}")
    print(f"  attention_mask[0]: {batch['attention_mask'][0].tolist()}")
    print(f"  attention_mask[0] sum: {batch['attention_mask'][0].sum().item()}")
    print(f"  mask[...,1:] sum:  {batch['attention_mask'][...,1:].sum().item()}")
    return dl

def check_3_weighted_ce(ds,tok,model):
    sep("CHECK 3: weighted_ce step-by-step")
    dev=next(model.parameters()).device
    col=WeightedLMCollator(tok)
    dl=make_ep_dl(ds,tok,col,bs=2,max_len=128)
    batch=next(iter(dl))
    batch={k:v.to(dev) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
    ids=batch["input_ids"]
    mask=batch["attention_mask"]
    w=batch["weights"]

    model.eval()
    with torch.no_grad():
        out=model(input_ids=ids,attention_mask=mask)

    logits=out.logits
    B,S,V=logits.shape
    sl=logits[...,:-1,:].contiguous().view(-1,V)
    tgt=ids[...,1:].contiguous().view(-1)
    loss_raw=CrossEntropyLoss(reduction="none")(sl,tgt).view(B,S-1)
    m=mask[...,1:].float()
    per=((loss_raw*m).sum(1))/(m.sum(1).clamp(min=1))
    result=(w*per).sum()/w.sum().clamp(min=1e-8)

    print(f"  logits shape: {logits.shape}")
    print(f"  m (mask shifted): shape={m.shape}  sum={m.sum().item()}")
    print(f"  m.sum(1) per sample: {m.sum(1).tolist()}")
    print(f"  loss_raw (first sample, first 5 pos): {loss_raw[0,:5].tolist()}")
    print(f"  per-sample loss: {per.tolist()}")
    print(f"  weights: {w.tolist()}")
    print(f"  weighted_ce result: {result.item()}")
    print(f"  --- Is weighted_ce zero? {result.item() == 0.0}")

def check_4_training_steps(ds,tok,model):
    sep("CHECK 4: 3 actual training steps")
    from torch.optim import AdamW
    dev=next(model.parameters()).device
    col=WeightedLMCollator(tok)
    dl=make_ep_dl(ds,tok,col,bs=2,max_len=128)
    opt=AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-4)
    model.train()
    for step,batch in enumerate(dl):
        if step>=3:break
        batch={k:v.to(dev) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
        ids=batch["input_ids"];mask=batch["attention_mask"];w=batch["weights"]
        out=model(input_ids=ids,attention_mask=mask)
        l_ep=weighted_ce(out.logits,ids,mask,w)
        print(f"  step {step}: l_ep={l_ep.item():.6f}  (zero={l_ep.item()==0.0})")
        opt.zero_grad()
        l_ep.backward()
        opt.step()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True)
    args=ap.parse_args()
    global MODEL_NAME
    MODEL_NAME=args.model

    sep(f"DEBUG ABLATION  model={MODEL_NAME}  cuda={torch.cuda.is_available()}")

    eps=make_episodes()
    store=EpisodeStore("/tmp/wm_debug.db")
    store.put_many(eps)
    chunks=Chunker(ChunkCfg(max_tok=256,overlap=32)).chunk_many(store.all())
    store.close()
    os.remove("/tmp/wm_debug.db") if os.path.exists("/tmp/wm_debug.db") else None
    print(f"chunks produced: {len(chunks)}")

    ds=check_1_dataset(chunks)

    print("\nLoading model (this takes a minute)...")
    model,tok=make_model(MODEL_NAME)
    model.print_trainable_parameters()

    check_2_dataloader(ds,tok)
    check_3_weighted_ce(ds,tok,model)
    check_4_training_steps(ds,tok,model)

    sep("DONE — share the output above to diagnose train_loss=0.0")

if __name__=="__main__":
    main()
