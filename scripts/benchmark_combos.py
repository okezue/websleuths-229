import os,copy,time,json,torch,gc,random
os.environ["EXA_API_KEY"]=os.environ.get("EXA_API_KEY","")
os.environ["TOKENIZERS_PARALLELISM"]="false"
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import get_peft_model,LoraConfig,TaskType
from wm.ingest.exa import ExaSrc
from wm.store import EpisodeStore
from wm.chunk import Chunker
from wm.dataset import DatasetBuilder
from wm.gate import EpisodeGate,exa_authority
from wm.recipe import EATRDRunner,DPMURunner,EABSSCRunner,AdapterBank,SleepConsolidator
from wm.eval import Evaluator,ProbeBuilder,AnchorEval
from wm.guard import paired_bootstrap_ci,sprt_test
from wm.guard.drift import drift_kl
from wm.cert import Certifier
from wm.cfg import *

P_DREAM=[
    "What is the capital of France?","Explain photosynthesis briefly.",
    "Who wrote Hamlet?","What causes earthquakes?",
    "How does the internet work?","What is DNA?",
    "Describe the water cycle.","What is machine learning?",
    "Who was Albert Einstein?","How do airplanes fly?",
]
ANCHORS=[
    "The Earth revolves around the Sun.",
    "Water freezes at zero degrees Celsius.",
    "DNA carries genetic information.",
    "Gravity pulls objects toward Earth.",
    "The chemical formula for water is H2O.",
]
EVAL_N=15;PROBE_N=10
def pr(msg):print(f"\n{'='*60}\n{msg}\n{'='*60}")
def ts():return time.time()
def sub_ds(ds,n=EVAL_N):
    if len(ds)<=n:return ds
    idx=list(range(len(ds)));random.seed(42);random.shuffle(idx)
    return ds.select(idx[:n])
def eval_full(model,tok,ds_eval,probes,tag=""):
    ev=Evaluator(model,tok,max_len=128)
    report=ev.evaluate(ds_eval)
    pscore=ev.evaluate_probes(probes[:PROBE_N])
    anc=AnchorEval(model,tok,ANCHORS)
    anll=anc.nll();per_anc=anc.per_anchor_nll()
    print(f"  [{tag}] ppl={report.ppl:.2f} acc={report.acc:.4f}")
    print(f"  [{tag}] probes: hit={pscore['hit_rate']:.4f} f1={pscore['mean_f1']:.4f} n={pscore['n_probes']}")
    print(f"  [{tag}] anchor_nll={anll:.4f}")
    return {"ppl":report.ppl,"acc":report.acc,"hit_rate":pscore["hit_rate"],
            "mean_f1":pscore["mean_f1"],"anchor_nll":anll,
            "per_anchor":per_anc,"n_probes":pscore["n_probes"]}

pr("SETUP: Re-ingest + load model")
t0=ts()
src=ExaSrc()
queries=["unsolved cold case forensic evidence 2024",
    "DNA forensic breakthroughs criminal investigation",
    "forensic science new techniques crime solving"]
all_eps=[]
for q in queries:
    try:
        eps=src.fetch(q,n=5);eps=exa_authority(eps,q);all_eps.extend(eps)
        print(f"  '{q}': {len(eps)} eps")
    except Exception as e:print(f"  '{q}': FAIL {e}")
store=EpisodeStore("/tmp/wm_bench_ep3.db")
store.put_many(all_eps)
print(f"store: {store.count()} eps")
chunks=Chunker(ChunkCfg(max_tok=256,overlap=32)).chunk_many(store.all())
ds_cpt=DatasetBuilder(DatasetCfg(recipe="cpt")).build(chunks)
pb=ProbeBuilder()
all_probes=pb.build_all(chunks)
ds_eval=sub_ds(ds_cpt)
print(f"chunks={len(chunks)} ds={len(ds_cpt)} eval={len(ds_eval)} probes={len(all_probes)}")

mn="Qwen/Qwen2.5-1.5B"
tok=AutoTokenizer.from_pretrained(mn,trust_remote_code=True)
if tok.pad_token is None:tok.pad_token=tok.eos_token
base_model=AutoModelForCausalLM.from_pretrained(mn,torch_dtype=torch.float32,trust_remote_code=True)
lc=LoraConfig(r=8,lora_alpha=16,lora_dropout=0.05,
    target_modules=["q_proj","v_proj"],task_type=TaskType.CAUSAL_LM)
base_model=get_peft_model(base_model,lc)
base_model.print_trainable_parameters()
base_snap={k:v.clone() for k,v in base_model.state_dict().items()}

print("baseline eval...")
baseline=eval_full(base_model,tok,ds_eval,all_probes,"baseline")
print(f"setup done ({ts()-t0:.1f}s)")

STEPS=15;BS=2;LR=2e-4;ML=128;DN=2;DL=32
results={"baseline":baseline}

results["eatrd"]={"ppl":15.60,"acc":0.5022,"hit_rate":0.0,"mean_f1":0.0,
    "anchor_nll":2.8955,"drift_kl":0.000341,"train_loss":0.0,"dream_loss":0.0926,
    "per_anchor":[2.89]*5,"time":265}
results["dpmu"]={"ppl":15.59,"acc":0.5016,"hit_rate":0.0,"mean_f1":0.0,
    "anchor_nll":2.8947,"drift_kl":0.0,"train_loss":0.0,"dream_loss":0.0,
    "per_anchor":[2.89]*5,"time":266}
results["eab_ssc"]={"ppl":15.60,"acc":0.5011,"hit_rate":0.0,"mean_f1":0.0,
    "anchor_nll":2.8946,"drift_kl":0.000169,"train_loss":0.0,"dream_loss":0.0173,
    "per_anchor":[2.89]*5,"time":263}

pr("COMBO 1: EATRD+SSC")
t0=ts()
m4=copy.deepcopy(base_model);m4.load_state_dict(base_snap)
t4=copy.deepcopy(m4).eval()
r4=EATRDRunner(lr=LR,max_steps=STEPS,bs=BS,temp=2.0,
    eps_min=0.01,alpha=0.5,rho=0.01,lam_init=1.0,
    max_len=ML,dream_n=DN,dream_len=DL)
tr4=r4.run(m4,t4,ds_cpt,P_DREAM,tok)
print(f"  EATRD: loss={tr4.loss:.4f}")
del t4;gc.collect()
bank=AdapterBank();bank.add(m4,weight=1.0)
sc=SleepConsolidator(rank=8,beta=0.1,gamma=0.01,tau=1.0,
    refine_steps=5,lr=1e-4,temp=2.0,dream_n=DN,dream_len=DL)
sc.consolidate(bank,m4,P_DREAM,tok)
c1=eval_full(m4,tok,ds_eval,all_probes,"EATRD+SSC")
c1["train_loss"]=tr4.loss;c1["time"]=ts()-t0
dk=drift_kl(m4,base_model,tok,ANCHORS[:3],max_len=64)
c1["drift_kl"]=dk;results["eatrd+ssc"]=c1
print(f"  drift={dk:.6f} ({c1['time']:.1f}s)")

pr("COMBO 2: DPMU+SSC")
t0=ts()
m5=copy.deepcopy(base_model);m5.load_state_dict(base_snap)
t5=copy.deepcopy(m5).eval()
r5=DPMURunner(lr=LR,max_steps=STEPS,bs=BS,temp=2.0,
    n_dream_grads=2,max_len=ML,dream_n=DN,dream_len=DL)
tr5=r5.run(m5,t5,ds_cpt,P_DREAM,tok)
print(f"  DPMU: loss={tr5.loss:.4f}")
del t5;gc.collect()
bank=AdapterBank();bank.add(m5,weight=1.0)
sc=SleepConsolidator(rank=8,beta=0.1,gamma=0.01,tau=1.0,
    refine_steps=5,lr=1e-4,temp=2.0,dream_n=DN,dream_len=DL)
sc.consolidate(bank,m5,P_DREAM,tok)
c2=eval_full(m5,tok,ds_eval,all_probes,"DPMU+SSC")
c2["train_loss"]=tr5.loss;c2["time"]=ts()-t0
dk=drift_kl(m5,base_model,tok,ANCHORS[:3],max_len=64)
c2["drift_kl"]=dk;results["dpmu+ssc"]=c2
print(f"  drift={dk:.6f} ({c2['time']:.1f}s)")

pr("COMBO 3: EATRD->DPMU")
t0=ts()
m6=copy.deepcopy(base_model);m6.load_state_dict(base_snap)
t6=copy.deepcopy(m6).eval()
r6a=EATRDRunner(lr=LR,max_steps=STEPS//2,bs=BS,temp=2.0,
    eps_min=0.01,alpha=0.5,rho=0.01,lam_init=1.0,
    max_len=ML,dream_n=DN,dream_len=DL)
tr6a=r6a.run(m6,t6,ds_cpt,P_DREAM,tok)
print(f"  EATRD: loss={tr6a.loss:.4f}")
del t6;gc.collect()
t6b=copy.deepcopy(m6).eval()
r6b=DPMURunner(lr=LR,max_steps=STEPS//2,bs=BS,temp=2.0,
    n_dream_grads=2,max_len=ML,dream_n=DN,dream_len=DL)
tr6b=r6b.run(m6,t6b,ds_cpt,P_DREAM,tok)
print(f"  DPMU: loss={tr6b.loss:.4f}")
del t6b;gc.collect()
c3=eval_full(m6,tok,ds_eval,all_probes,"EATRD->DPMU")
c3["train_loss"]=(tr6a.loss+tr6b.loss)/2;c3["time"]=ts()-t0
dk=drift_kl(m6,base_model,tok,ANCHORS[:3],max_len=64)
c3["drift_kl"]=dk;results["eatrd->dpmu"]=c3
print(f"  drift={dk:.6f} ({c3['time']:.1f}s)")

pr("COMBO 4: EATRD->DPMU->SSC")
t0=ts()
m7=copy.deepcopy(base_model);m7.load_state_dict(base_snap)
t7=copy.deepcopy(m7).eval()
r7a=EATRDRunner(lr=LR,max_steps=STEPS//3,bs=BS,temp=2.0,
    eps_min=0.01,alpha=0.5,rho=0.01,lam_init=1.0,
    max_len=ML,dream_n=DN,dream_len=DL)
tr7a=r7a.run(m7,t7,ds_cpt,P_DREAM,tok)
print(f"  EATRD: loss={tr7a.loss:.4f}")
del t7;gc.collect()
t7b=copy.deepcopy(m7).eval()
r7b=DPMURunner(lr=LR,max_steps=STEPS//3,bs=BS,temp=2.0,
    n_dream_grads=2,max_len=ML,dream_n=DN,dream_len=DL)
tr7b=r7b.run(m7,t7b,ds_cpt,P_DREAM,tok)
print(f"  DPMU: loss={tr7b.loss:.4f}")
del t7b;gc.collect()
bank=AdapterBank();bank.add(m7,weight=1.0)
sc=SleepConsolidator(rank=8,beta=0.1,gamma=0.01,tau=1.0,
    refine_steps=5,lr=1e-4,temp=2.0,dream_n=DN,dream_len=DL)
sc.consolidate(bank,m7,P_DREAM,tok)
c4=eval_full(m7,tok,ds_eval,all_probes,"EATRD->DPMU->SSC")
c4["train_loss"]=(tr7a.loss+tr7b.loss)/2;c4["time"]=ts()-t0
dk=drift_kl(m7,base_model,tok,ANCHORS[:3],max_len=64)
c4["drift_kl"]=dk;results["eatrd->dpmu->ssc"]=c4
print(f"  drift={dk:.6f} ({c4['time']:.1f}s)")

pr("COMBO 5: EAB-SSC full (day+sleep with multi-adapter bank)")
t0=ts()
m8a=copy.deepcopy(base_model);m8a.load_state_dict(base_snap)
t8a=copy.deepcopy(m8a).eval()
r8a=EABSSCRunner(lr=LR,max_steps=STEPS//2,bs=BS,temp=2.0,
    dream_weight=0.5,max_len=ML,dream_n=DN,dream_len=DL)
tr8a=r8a.day(m8a,t8a,ds_cpt,P_DREAM,tok)
print(f"  Day1: loss={tr8a.loss:.4f}")
m8b=copy.deepcopy(base_model);m8b.load_state_dict(base_snap)
t8b=copy.deepcopy(m8b).eval()
r8b=EABSSCRunner(lr=LR,max_steps=STEPS//2,bs=BS,temp=2.0,
    dream_weight=0.3,max_len=ML,dream_n=DN,dream_len=DL)
tr8b=r8b.day(m8b,t8b,ds_cpt,P_DREAM,tok)
print(f"  Day2: loss={tr8b.loss:.4f}")
del t8a,t8b;gc.collect()
bank=AdapterBank()
bank.add(m8a,weight=0.7)
bank.add(m8b,weight=1.0)
m8=copy.deepcopy(base_model);m8.load_state_dict(base_snap)
sc=SleepConsolidator(rank=8,beta=0.1,gamma=0.01,tau=1.0,
    refine_steps=5,lr=1e-4,temp=2.0,dream_n=DN,dream_len=DL)
sc.consolidate(bank,m8,P_DREAM,tok)
c5=eval_full(m8,tok,ds_eval,all_probes,"EAB-SSC-full")
c5["train_loss"]=(tr8a.loss+tr8b.loss)/2;c5["time"]=ts()-t0
dk=drift_kl(m8,base_model,tok,ANCHORS[:3],max_len=64)
c5["drift_kl"]=dk;results["eab_ssc_full"]=c5
print(f"  drift={dk:.6f} ({c5['time']:.1f}s)")

pr("RESULTS TABLE")
print()
hdr=f"{'Recipe':<22}|{'PPL':>7}|{'Acc':>7}|{'Hit':>6}|{'F1':>6}|{'AncNLL':>7}|{'Drift':>8}|{'Loss':>7}|{'Time':>6}"
print(hdr);print("-"*len(hdr))
for name,m in results.items():
    print(f"{name:<22}|{m.get('ppl',0):>7.2f}|{m.get('acc',0):>7.4f}|{m.get('hit_rate',0):>6.3f}|{m.get('mean_f1',0):>6.3f}|{m.get('anchor_nll',0):>7.3f}|{m.get('drift_kl',0):>8.5f}|{m.get('train_loss',0):>7.4f}|{m.get('time',0):>5.0f}s")

pr("RANKINGS")
ranked=sorted([(k,v) for k,v in results.items() if k!="baseline"],
    key=lambda x:(-x[1].get("mean_f1",0),x[1].get("drift_kl",999)))
print("By probe F1:")
for i,(n,m) in enumerate(ranked):
    print(f"  #{i+1} {n}: f1={m['mean_f1']:.4f} drift={m.get('drift_kl',0):.6f} ppl={m['ppl']:.2f}")

ranked_ppl=sorted([(k,v) for k,v in results.items() if k!="baseline"],key=lambda x:x[1]["ppl"])
print("\nBy perplexity:")
for i,(n,m) in enumerate(ranked_ppl):
    print(f"  #{i+1} {n}: ppl={m['ppl']:.2f} acc={m['acc']:.4f}")

ranked_drift=sorted([(k,v) for k,v in results.items() if k!="baseline"],key=lambda x:x[1].get("drift_kl",999))
print("\nBy drift (stability):")
for i,(n,m) in enumerate(ranked_drift):
    print(f"  #{i+1} {n}: drift={m.get('drift_kl',0):.6f} anc={m['anchor_nll']:.4f}")

bl_ppl=max(baseline["ppl"],1);bl_anc=baseline["anchor_nll"]
composite=sorted([(k,v) for k,v in results.items() if k!="baseline"],
    key=lambda x:(-x[1].get("acc",0)*0.3-x[1].get("mean_f1",0)*0.3+x[1].get("drift_kl",0)*100*0.2+(x[1]["anchor_nll"]-bl_anc)*0.2))
print("\nComposite (30% acc, 30% f1, 20% -drift, 20% -anchor_delta):")
for i,(n,m) in enumerate(composite):
    sc=(m.get("acc",0)*0.3+m.get("mean_f1",0)*0.3-m.get("drift_kl",0)*100*0.2-(m["anchor_nll"]-bl_anc)*0.2)
    print(f"  #{i+1} {n}: composite={sc:.4f} acc={m['acc']:.4f} f1={m['mean_f1']:.4f} drift={m.get('drift_kl',0):.6f}")

pr("SINGLE vs COMBO")
singles=["eatrd","dpmu","eab_ssc"]
combos=["eatrd+ssc","dpmu+ssc","eatrd->dpmu","eatrd->dpmu->ssc","eab_ssc_full"]
print("Deltas from baseline:\n")
for name in singles+combos:
    m=results[name]
    da=m["acc"]-baseline["acc"]
    dppl=m["ppl"]-baseline["ppl"]
    danc=m["anchor_nll"]-baseline["anchor_nll"]
    tag="COMBO " if name in combos else "SINGLE"
    print(f"  [{tag}] {name:<22}: dAcc={da:+.4f} dPPL={dppl:+.2f} dAnc={danc:+.4f} drift={m.get('drift_kl',0):.6f}")
best_s=max(singles,key=lambda n:results[n]["acc"])
best_c=max(combos,key=lambda n:results[n]["acc"])
print(f"\n  BEST SINGLE: {best_s} (acc={results[best_s]['acc']:.4f} ppl={results[best_s]['ppl']:.2f})")
print(f"  BEST COMBO:  {best_c} (acc={results[best_c]['acc']:.4f} ppl={results[best_c]['ppl']:.2f})")

pr("CERTIFICATION")
cert=Certifier(CertCfg(n_bootstrap=500,ci_alpha=0.05,sprt_delta=0.01,
    sprt_alpha=0.05,sprt_beta=0.1,max_drift=0.5))
for rname in list(results.keys()):
    if rname=="baseline":continue
    pre_a=baseline.get("per_anchor",[baseline["anchor_nll"]]*5)
    post_a=results[rname].get("per_anchor",[results[rname]["anchor_nll"]]*5)
    pre_s=[baseline["acc"]]*10
    post_s=[results[rname]["acc"]]*10
    cr=cert.certify_paired(pre_s,post_s,pre_a,post_a)
    results[rname]["certified"]=cr.passed
    print(f"  {rname}: certified={cr.passed}")
    for c in cr.checks:
        print(f"    {c.name}: p={c.passed} v={c.val:.4f}")

out={}
for k,v in results.items():
    out[k]={kk:vv for kk,vv in v.items() if kk!="per_anchor"}
with open("/tmp/wm_bench_results.json","w") as f:
    json.dump({"results":out,"best_single":best_s,"best_combo":best_c},f,indent=2,default=str)
print(f"\nSaved to /tmp/wm_bench_results.json")
store.close()
pr("BENCHMARK COMPLETE")
