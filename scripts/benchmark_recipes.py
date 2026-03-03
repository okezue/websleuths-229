import os,copy,time,json,torch,gc,random
os.environ["EXA_API_KEY"]="e337f35a-e56c-4ae7-8596-f44959053342"
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import get_peft_model,LoraConfig,TaskType
from datasets import Dataset
from wm.ingest.exa import ExaSrc
from wm.store import EpisodeStore
from wm.chunk import Chunker
from wm.dataset import DatasetBuilder
from wm.gate import EpisodeGate,exa_authority
from wm.recipe import EATRDRunner,DPMURunner,EABSSCRunner,AdapterBank,SleepConsolidator
from wm.eval import Evaluator,ProbeBuilder,AnchorEval
from wm.guard import UpdateGuard,expected_gain,paired_bootstrap_ci,sprt_test
from wm.guard.drift import drift_kl
from wm.cert import Certifier
from wm.bench.compare import RecipeBenchmark,BenchResult
from wm.cfg import *
from wm.types import Chunk

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
EVAL_N=15
PROBE_N=10
def pr(msg):print(f"\n{'='*60}\n{msg}\n{'='*60}")
def ts():return time.time()
def sub_ds(ds,n=EVAL_N):
    if len(ds)<=n:return ds
    idx=list(range(len(ds)))
    random.seed(42)
    random.shuffle(idx)
    return ds.select(idx[:n])

def make_model(mn="Qwen/Qwen2.5-1.5B",r=8,alpha=16):
    tok=AutoTokenizer.from_pretrained(mn,trust_remote_code=True)
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(mn,torch_dtype=torch.float32,trust_remote_code=True)
    lc=LoraConfig(r=r,lora_alpha=alpha,lora_dropout=0.05,
        target_modules=["q_proj","v_proj"],task_type=TaskType.CAUSAL_LM)
    model=get_peft_model(model,lc)
    return model,tok

def eval_full(model,tok,ds_eval,probes,tag=""):
    ev=Evaluator(model,tok,max_len=128)
    report=ev.evaluate(ds_eval)
    pscore=ev.evaluate_probes(probes[:PROBE_N])
    anc=AnchorEval(model,tok,ANCHORS)
    anll=anc.nll()
    per_anc=anc.per_anchor_nll()
    print(f"  [{tag}] ppl={report.ppl:.2f} acc={report.acc:.4f}")
    print(f"  [{tag}] probes: hit={pscore['hit_rate']:.4f} f1={pscore['mean_f1']:.4f} n={pscore['n_probes']}")
    print(f"  [{tag}] anchor_nll={anll:.4f}")
    return {"ppl":report.ppl,"acc":report.acc,"hit_rate":pscore["hit_rate"],
            "mean_f1":pscore["mean_f1"],"anchor_nll":anll,
            "per_anchor":per_anc,"n_probes":pscore["n_probes"]}

pr("PHASE 1: INGEST FROM EXA (real web data)")
t0=ts()
src=ExaSrc()
queries=[
    "unsolved cold case forensic evidence 2024",
    "DNA forensic breakthroughs criminal investigation",
    "forensic science new techniques crime solving",
]
all_eps=[]
for q in queries:
    try:
        eps=src.fetch(q,n=5)
        eps=exa_authority(eps)
        all_eps.extend(eps)
        print(f"  '{q}': {len(eps)} eps")
    except Exception as e:
        print(f"  '{q}': FAILED - {e}")
gate=EpisodeGate(GateCfg(min_sources=1,min_consistency=0.0,uncertainty_thresh=0.0))
gr=gate.check(all_eps)
print(f"gate: accept={gr.accept} n={gr.n_sources} cons={gr.consistency:.3f}")
for e in all_eps[:5]:
    print(f"  [{e.eid[:8]}] auth={e.authority:.3f} '{e.title[:70]}'")
store=EpisodeStore("/tmp/wm_bench_ep2.db")
store.put_many(all_eps)
print(f"store: {store.count()} eps ({ts()-t0:.1f}s)")

pr("PHASE 2: CHUNK + DATASET")
chunks=Chunker(ChunkCfg(max_tok=256,overlap=32)).chunk_many(store.all())
print(f"chunks: {len(chunks)}")
ds_cpt=DatasetBuilder(DatasetCfg(recipe="cpt")).build(chunks)
ds_sft=DatasetBuilder(DatasetCfg(recipe="ext_sft")).build(chunks)
ds_qa=DatasetBuilder(DatasetCfg(recipe="cited_qa")).build(chunks)
print(f"cpt={len(ds_cpt)} sft={len(ds_sft)} qa={len(ds_qa)}")
pb=ProbeBuilder()
all_probes=pb.build_all(chunks)
print(f"probes: {len(all_probes)} (cloze={sum(1 for p in all_probes if p.kind=='cloze')}, qa={sum(1 for p in all_probes if p.kind=='qa')})")
ds_eval=sub_ds(ds_cpt)
print(f"eval subset: {len(ds_eval)} rows")

pr("PHASE 3: LOAD Qwen2.5-1.5B + BASELINE")
t0=ts()
base_model,tok=make_model()
base_model.print_trainable_parameters()
print(f"load: {ts()-t0:.1f}s")
base_snap={k:v.clone() for k,v in base_model.state_dict().items()}
print("computing baseline...")
baseline=eval_full(base_model,tok,ds_eval,all_probes,"baseline")
print(f"baseline done ({ts()-t0:.1f}s)")

STEPS=15
BS=2
LR=2e-4
ML=128
DN=2
DL=32
results={}
results["baseline"]=baseline

pr("PHASE 4a: EATRD")
t0=ts()
m1=copy.deepcopy(base_model)
m1.load_state_dict(base_snap)
teacher1=copy.deepcopy(m1).eval()
r1=EATRDRunner(lr=LR,max_steps=STEPS,bs=BS,temp=2.0,
    eps_min=0.01,alpha=0.5,rho=0.01,lam_init=1.0,
    max_len=ML,dream_n=DN,dream_len=DL)
tr1=r1.run(m1,teacher1,ds_cpt,P_DREAM,tok)
print(f"  train: loss={tr1.loss:.4f} dream={tr1.dream_loss:.4f} steps={tr1.steps}")
print(f"  lam={tr1.extras.get('lambda',0):.4f} eps_k={tr1.extras.get('eps_k',0):.4f}")
del teacher1;gc.collect()
eatrd_m=eval_full(m1,tok,ds_eval,all_probes,"EATRD")
eatrd_m["train_loss"]=tr1.loss
eatrd_m["dream_loss"]=tr1.dream_loss
eatrd_m["time"]=ts()-t0
dk1=drift_kl(m1,base_model,tok,ANCHORS[:3],max_len=64)
eatrd_m["drift_kl"]=dk1
results["eatrd"]=eatrd_m
print(f"  drift_kl={dk1:.6f} ({eatrd_m['time']:.1f}s)")

pr("PHASE 4b: DPMU")
t0=ts()
m2=copy.deepcopy(base_model)
m2.load_state_dict(base_snap)
teacher2=copy.deepcopy(m2).eval()
r2=DPMURunner(lr=LR,max_steps=STEPS,bs=BS,temp=2.0,
    n_dream_grads=2,max_len=ML,dream_n=DN,dream_len=DL)
tr2=r2.run(m2,teacher2,ds_cpt,P_DREAM,tok)
print(f"  train: loss={tr2.loss:.4f} dream={tr2.dream_loss:.4f} steps={tr2.steps}")
del teacher2;gc.collect()
dpmu_m=eval_full(m2,tok,ds_eval,all_probes,"DPMU")
dpmu_m["train_loss"]=tr2.loss
dpmu_m["dream_loss"]=tr2.dream_loss
dpmu_m["time"]=ts()-t0
dk2=drift_kl(m2,base_model,tok,ANCHORS[:3],max_len=64)
dpmu_m["drift_kl"]=dk2
results["dpmu"]=dpmu_m
print(f"  drift_kl={dk2:.6f} ({dpmu_m['time']:.1f}s)")

pr("PHASE 4c: EAB-SSC")
t0=ts()
m3=copy.deepcopy(base_model)
m3.load_state_dict(base_snap)
teacher3=copy.deepcopy(m3).eval()
r3=EABSSCRunner(lr=LR,max_steps=STEPS,bs=BS,temp=2.0,
    dream_weight=0.5,max_len=ML,dream_n=DN,dream_len=DL)
tr3=r3.day(m3,teacher3,ds_cpt,P_DREAM,tok)
print(f"  train: loss={tr3.loss:.4f} dream={tr3.dream_loss:.4f} steps={tr3.steps}")
del teacher3;gc.collect()
eab_m=eval_full(m3,tok,ds_eval,all_probes,"EAB-SSC")
eab_m["train_loss"]=tr3.loss
eab_m["dream_loss"]=tr3.dream_loss
eab_m["time"]=ts()-t0
dk3=drift_kl(m3,base_model,tok,ANCHORS[:3],max_len=64)
eab_m["drift_kl"]=dk3
results["eab_ssc"]=eab_m
print(f"  drift_kl={dk3:.6f} ({eab_m['time']:.1f}s)")

pr("PHASE 5: UPDATEGUARD + CERTIFICATION")
for rname in ["eatrd","dpmu","eab_ssc"]:
    pre_s=[baseline["mean_f1"]]*10
    post_s=[results[rname]["mean_f1"]]*10
    pre_a=baseline["per_anchor"]
    post_a=results[rname]["per_anchor"]
    ci_lo,ci_hi=paired_bootstrap_ci(pre_s,post_s,n_boot=500,alpha=0.05)
    sv=sprt_test(pre_s,post_s,delta=0.01)
    print(f"  {rname}: CI=[{ci_lo:.4f},{ci_hi:.4f}] SPRT={sv}")
    results[rname]["ci_lo"]=ci_lo
    results[rname]["ci_hi"]=ci_hi
    results[rname]["sprt"]=sv
    cert=Certifier(CertCfg(n_bootstrap=500,ci_alpha=0.05,sprt_delta=0.01,
        sprt_alpha=0.05,sprt_beta=0.1,max_drift=0.5))
    cr=cert.certify_paired(pre_s,post_s,pre_a,post_a)
    print(f"  {rname}: certified={cr.passed}")
    for c in cr.checks:
        print(f"    {c.name}: p={c.passed} v={c.val:.4f} t={c.thresh:.4f}")
    results[rname]["certified"]=cr.passed

pr("PHASE 6: COMBINATIONS")
pr("6a: EATRD+SSC")
t0=ts()
m4=copy.deepcopy(base_model);m4.load_state_dict(base_snap)
teacher4=copy.deepcopy(m4).eval()
r4a=EATRDRunner(lr=LR,max_steps=STEPS,bs=BS,temp=2.0,
    eps_min=0.01,alpha=0.5,rho=0.01,lam_init=1.0,
    max_len=ML,dream_n=DN,dream_len=DL)
tr4a=r4a.run(m4,teacher4,ds_cpt,P_DREAM,tok)
print(f"  EATRD: loss={tr4a.loss:.4f}")
del teacher4;gc.collect()
bank4=AdapterBank();bank4.add(m4,weight=1.0)
sc4=SleepConsolidator(rank=8,beta=0.1,gamma=0.01,tau=1.0,
    refine_steps=5,lr=1e-4,temp=2.0,dream_n=DN,dream_len=DL)
sc4.consolidate(bank4,m4,P_DREAM,tok)
c1=eval_full(m4,tok,ds_eval,all_probes,"EATRD+SSC")
c1["train_loss"]=tr4a.loss;c1["time"]=ts()-t0
dk4=drift_kl(m4,base_model,tok,ANCHORS[:3],max_len=64)
c1["drift_kl"]=dk4;results["eatrd+ssc"]=c1
print(f"  drift={dk4:.6f} ({c1['time']:.1f}s)")

pr("6b: DPMU+SSC")
t0=ts()
m5=copy.deepcopy(base_model);m5.load_state_dict(base_snap)
teacher5=copy.deepcopy(m5).eval()
r5a=DPMURunner(lr=LR,max_steps=STEPS,bs=BS,temp=2.0,
    n_dream_grads=2,max_len=ML,dream_n=DN,dream_len=DL)
tr5a=r5a.run(m5,teacher5,ds_cpt,P_DREAM,tok)
print(f"  DPMU: loss={tr5a.loss:.4f}")
del teacher5;gc.collect()
bank5=AdapterBank();bank5.add(m5,weight=1.0)
sc5=SleepConsolidator(rank=8,beta=0.1,gamma=0.01,tau=1.0,
    refine_steps=5,lr=1e-4,temp=2.0,dream_n=DN,dream_len=DL)
sc5.consolidate(bank5,m5,P_DREAM,tok)
c2=eval_full(m5,tok,ds_eval,all_probes,"DPMU+SSC")
c2["train_loss"]=tr5a.loss;c2["time"]=ts()-t0
dk5=drift_kl(m5,base_model,tok,ANCHORS[:3],max_len=64)
c2["drift_kl"]=dk5;results["dpmu+ssc"]=c2
print(f"  drift={dk5:.6f} ({c2['time']:.1f}s)")

pr("6c: EATRD->DPMU")
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
dk6=drift_kl(m6,base_model,tok,ANCHORS[:3],max_len=64)
c3["drift_kl"]=dk6;results["eatrd->dpmu"]=c3
print(f"  drift={dk6:.6f} ({c3['time']:.1f}s)")

pr("6d: EATRD->DPMU->SSC (full pipeline)")
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
bank7=AdapterBank();bank7.add(m7,weight=1.0)
sc7=SleepConsolidator(rank=8,beta=0.1,gamma=0.01,tau=1.0,
    refine_steps=5,lr=1e-4,temp=2.0,dream_n=DN,dream_len=DL)
sc7.consolidate(bank7,m7,P_DREAM,tok)
c4=eval_full(m7,tok,ds_eval,all_probes,"EATRD->DPMU->SSC")
c4["train_loss"]=(tr7a.loss+tr7b.loss)/2;c4["time"]=ts()-t0
dk7=drift_kl(m7,base_model,tok,ANCHORS[:3],max_len=64)
c4["drift_kl"]=dk7;results["eatrd->dpmu->ssc"]=c4
print(f"  drift={dk7:.6f} ({c4['time']:.1f}s)")

pr("PHASE 7: RESULTS TABLE")
print()
hdr=f"{'Recipe':<22}|{'PPL':>7}|{'Acc':>7}|{'Hit':>6}|{'F1':>6}|{'AncNLL':>7}|{'Drift':>8}|{'Loss':>7}|{'Time':>6}"
print(hdr)
print("-"*len(hdr))
for name,m in results.items():
    print(f"{name:<22}|{m.get('ppl',0):>7.2f}|{m.get('acc',0):>7.4f}|{m.get('hit_rate',0):>6.3f}|{m.get('mean_f1',0):>6.3f}|{m.get('anchor_nll',0):>7.3f}|{m.get('drift_kl',0):>8.5f}|{m.get('train_loss',0):>7.4f}|{m.get('time',0):>5.0f}s")

pr("PHASE 8: RANKINGS")
ranked=sorted(
    [(k,v) for k,v in results.items() if k!="baseline"],
    key=lambda x:(-x[1].get("mean_f1",0),x[1].get("drift_kl",999)))
print("By probe F1 (tiebreak: drift):")
for i,(n,m) in enumerate(ranked):
    ct=" CERT" if m.get("certified") else ""
    print(f"  #{i+1} {n}: f1={m['mean_f1']:.4f} drift={m.get('drift_kl',0):.6f} ppl={m['ppl']:.2f}{ct}")

ranked_ppl=sorted([(k,v) for k,v in results.items() if k!="baseline"],key=lambda x:x[1]["ppl"])
print("\nBy perplexity:")
for i,(n,m) in enumerate(ranked_ppl):
    print(f"  #{i+1} {n}: ppl={m['ppl']:.2f} acc={m['acc']:.4f}")

ranked_drift=sorted([(k,v) for k,v in results.items() if k!="baseline"],key=lambda x:x[1].get("drift_kl",999))
print("\nBy drift (stability):")
for i,(n,m) in enumerate(ranked_drift):
    print(f"  #{i+1} {n}: drift={m.get('drift_kl',0):.6f} anc={m['anchor_nll']:.4f}")

pr("PHASE 9: COMPOSITE RANKING")
bl_ppl=max(baseline["ppl"],1)
bl_anc=baseline["anchor_nll"]
composite=sorted(
    [(k,v) for k,v in results.items() if k!="baseline"],
    key=lambda x:(
        -x[1].get("mean_f1",0)*0.4
        +(x[1]["ppl"]/bl_ppl)*0.2
        +x[1].get("drift_kl",0)*100*0.2
        +(x[1]["anchor_nll"]-bl_anc)*0.2))
for i,(n,m) in enumerate(composite):
    sc=(m.get("mean_f1",0)*0.4
        -(m["ppl"]/bl_ppl-1)*0.2
        -m.get("drift_kl",0)*100*0.2
        -(m["anchor_nll"]-bl_anc)*0.2)
    print(f"  #{i+1} {n}: composite={sc:.4f}")

pr("PHASE 10: SINGLE vs COMBO ANALYSIS")
singles=["eatrd","dpmu","eab_ssc"]
combos=["eatrd+ssc","dpmu+ssc","eatrd->dpmu","eatrd->dpmu->ssc"]
print("Deltas from baseline:\n")
for name in singles+combos:
    m=results[name]
    df1=m["mean_f1"]-baseline["mean_f1"]
    dppl=m["ppl"]-baseline["ppl"]
    danc=m["anchor_nll"]-baseline["anchor_nll"]
    tag="COMBO " if "+" in name or "->" in name else "SINGLE"
    print(f"  [{tag}] {name:<22}: dF1={df1:+.4f} dPPL={dppl:+.2f} dAnc={danc:+.4f} drift={m.get('drift_kl',0):.6f}")
best_s=max(singles,key=lambda n:results[n]["mean_f1"])
best_c=max(combos,key=lambda n:results[n]["mean_f1"])
print(f"\n  BEST SINGLE: {best_s} (f1={results[best_s]['mean_f1']:.4f})")
print(f"  BEST COMBO:  {best_c} (f1={results[best_c]['mean_f1']:.4f})")
print(f"  Combo advantage: {results[best_c]['mean_f1']-results[best_s]['mean_f1']:+.4f} F1")

out={}
for k,v in results.items():
    out[k]={kk:vv for kk,vv in v.items() if kk!="per_anchor"}
with open("/tmp/wm_bench_results.json","w") as f:
    json.dump({"results":out,"best_single":best_s,"best_combo":best_c},f,indent=2,default=str)
print(f"\nSaved to /tmp/wm_bench_results.json")
store.close()
pr("BENCHMARK COMPLETE")
