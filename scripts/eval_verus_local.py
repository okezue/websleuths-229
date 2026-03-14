import json,sys,os,time,torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel

VPATH=os.environ.get("VERUS_DATA","/Users/okezuebell/Documents/GitHub/VerusFT_RL/datasets_training/task_a_tasks_v9.jsonl")
N=50

def gen(m,t,prompt,max_tok=200):
    enc=t(prompt,return_tensors="pt",truncation=True,max_length=512)
    inp={k:v.to(m.device) for k,v in enc.items() if k in ("input_ids","attention_mask")}
    with torch.no_grad():
        out=m.generate(**inp,max_new_tokens=max_tok,do_sample=False,temperature=1.0)
    return t.decode(out[0][inp["input_ids"].shape[1]:],skip_special_tokens=True)

def eval_verus(m,t,rows,n):
    import random
    random.seed(42)
    idx=random.sample(range(len(rows)),min(n,len(rows)))
    cor,tot=0,0
    results=[]
    for i in idx:
        row=rows[i]
        inp=row.get("input_text","")[:500]
        tgt=row.get("target_text","")
        prompt=f"Given this Rust/Verus function, generate the requires/ensures specifications:\n\n{inp}\n\nSpecifications:"
        g=gen(m,t,prompt,max_tok=200)
        tgt_keys=[kw for kw in ["requires","ensures","invariant","decreases"] if kw in tgt.lower()]
        gen_keys=[kw for kw in ["requires","ensures","invariant","decreases"] if kw in g.lower()]
        overlap=len(set(tgt_keys)&set(gen_keys))
        total=max(len(set(tgt_keys)),1)
        hit=overlap/total>=0.5
        if hit:cor+=1
        tot+=1
        results.append({"id":row.get("id",""),"input":inp[:100],"target_keys":tgt_keys,"gen_keys":gen_keys,"hit":hit,"gen":g[:300]})
        if tot%10==0:print(f"  {tot}/{n} done, acc={cor/tot:.3f}")
    return cor/max(tot,1),tot,results

def eval_minif2f(m,t,n=50):
    try:
        from datasets import load_dataset
        ds=load_dataset("cat-searcher/minif2f-lean4",split="test",trust_remote_code=True)
    except:
        try:
            ds=load_dataset("cat-searcher/minif2f-lean4",split="validation",trust_remote_code=True)
        except:
            print("minif2f dataset not loadable")
            return 0.0,0,[]
    import random
    random.seed(42)
    idx=random.sample(range(len(ds)),min(n,len(ds)))
    cor,tot=0,0
    results=[]
    for i in idx:
        row=ds[i]
        stmt=row.get("formal_statement","") or row.get("statement","") or row.get("problem","") or ""
        if not stmt:tot+=1;continue
        prompt=f"Complete this Lean 4 theorem proof:\n\n{stmt[:400]}\n\nProof:"
        g=gen(m,t,prompt,max_tok=150)
        has_proof=any(kw in g.lower() for kw in ["sorry","by","simp","ring","omega","linarith","norm_num","exact","apply","intro","have"])
        if has_proof:cor+=1
        tot+=1
        results.append({"stmt":stmt[:100],"gen":g[:200],"hit":has_proof})
    return cor/max(tot,1),tot,results

rows=[]
with open(VPATH) as f:
    for line in f:
        if line.strip():rows.append(json.loads(line))
print(f"Loaded {len(rows)} Verus tasks")

base_name="Qwen/Qwen2.5-1.5B"
print(f"\nLoading {base_name}...")
tok=AutoTokenizer.from_pretrained(base_name,trust_remote_code=True)
if tok.pad_token is None:tok.pad_token=tok.eos_token
m=AutoModelForCausalLM.from_pretrained(base_name,torch_dtype=torch.float32,trust_remote_code=True)
m.eval()

out={"model":base_name,"verus_data":VPATH,"n":N,"checkpoints":{}}

print(f"\n=== Baseline {base_name} ===")
t0=time.time()
acc,tot,res=eval_verus(m,tok,rows,N)
print(f"Verus: acc={acc:.4f} n={tot} time={time.time()-t0:.1f}s")
out["checkpoints"]["baseline"]={"verus_acc":acc,"verus_n":tot,"verus_results":res}

t0=time.time()
m2f_acc,m2f_n,m2f_res=eval_minif2f(m,tok,N)
print(f"miniF2F: acc={m2f_acc:.4f} n={m2f_n} time={time.time()-t0:.1f}s")
out["checkpoints"]["baseline"]["minif2f_acc"]=m2f_acc
out["checkpoints"]["baseline"]["minif2f_n"]=m2f_n
out["checkpoints"]["baseline"]["minif2f_results"]=m2f_res

ckpt_dir="/Users/okezuebell/Documents/GitHub/websleuths-229/results/dreaming_bench/checkpoints"
for dom in ["finance","legal","chemistry","medicine"]:
    cp=os.path.join(ckpt_dir,dom)
    if not os.path.exists(os.path.join(cp,"adapter_config.json")):
        print(f"\nSkipping {dom} (no checkpoint)")
        continue
    print(f"\n=== After {dom} training ===")
    base=AutoModelForCausalLM.from_pretrained(base_name,torch_dtype=torch.float32,trust_remote_code=True)
    pm=PeftModel.from_pretrained(base,cp)
    pm.eval()
    t0=time.time()
    acc,tot,res=eval_verus(pm,tok,rows,N)
    print(f"Verus: acc={acc:.4f} n={tot} time={time.time()-t0:.1f}s")
    out["checkpoints"][dom]={"verus_acc":acc,"verus_n":tot,"verus_results":res}
    t0=time.time()
    m2f_acc,m2f_n,m2f_res=eval_minif2f(pm,tok,N)
    print(f"miniF2F: acc={m2f_acc:.4f} n={m2f_n} time={time.time()-t0:.1f}s")
    out["checkpoints"][dom]["minif2f_acc"]=m2f_acc
    out["checkpoints"][dom]["minif2f_n"]=m2f_n
    out["checkpoints"][dom]["minif2f_results"]=m2f_res
    del base,pm
    import gc;gc.collect()

opath="/Users/okezuebell/Documents/GitHub/websleuths-229/results/verus_eval_qwen.json"
with open(opath,"w") as f:json.dump(out,f,indent=2)
print(f"\nSaved to {opath}")
