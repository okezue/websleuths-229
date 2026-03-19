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
from wm.guard import UpdateGuard,paired_bootstrap_ci,sprt_test
from wm.guard.drift import drift_kl
from wm.cert import Certifier
from wm.cfg import *

P_DREAM=[
    "What is the capital of France?","Explain photosynthesis briefly.",
    "Who wrote Hamlet?","What causes earthquakes?",
    "How does the internet work?","What is DNA?",
    "Describe the water cycle.","What is machine learning?",
    "Who was Albert Einstein?","How do airplanes fly?",
    "What is the speed of light?","How do vaccines work?",
    "What causes the seasons?","What is evolution?",
    "How does electricity work?","What is the periodic table?",
    "Who discovered gravity?","What is climate change?",
    "How do computers store data?","What is the solar system?",
]
ANCHORS=[
    "The Earth revolves around the Sun.",
    "Water freezes at zero degrees Celsius.",
    "DNA carries genetic information in living organisms.",
    "Gravity pulls objects toward the center of the Earth.",
    "The chemical formula for water is H2O.",
    "Oxygen is essential for human respiration.",
    "The Pacific Ocean is the largest ocean on Earth.",
    "Iron is a magnetic metal.",
    "Nitrogen makes up about 78 percent of the atmosphere.",
    "Light travels in straight lines.",
]
EVAL_N=30
PROBE_N=20
STEPS=60
BS=2
LR=3e-4
ML=256
DN=4
DL=64

def pr(msg):print(f"\n{'='*70}\n{msg}\n{'='*70}")
def ts():return time.time()
def sub_ds(ds,n=EVAL_N):
    if len(ds)<=n:return ds
    idx=list(range(len(ds)));random.seed(42);random.shuffle(idx)
    return ds.select(idx[:n])

def eval_full(model,tok,ds_eval,probes,tag=""):
    ev=Evaluator(model,tok,max_len=ML)
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

pr("PHASE 1: INGEST — broader queries for more diverse data")
t0=ts()
src=ExaSrc()
queries=[
    "unsolved cold case forensic evidence 2024",
    "DNA forensic breakthroughs criminal investigation",
    "forensic science new techniques crime solving",
    "cold case solved genetic genealogy",
    "forensic pathology murder investigation evidence",
]
all_eps=[]
for q in queries:
    try:
        eps=src.fetch(q,n=10)
        eps=exa_authority(eps,q)
        all_eps.extend(eps)
        print(f"  '{q[:50]}': {len(eps)} eps")
    except Exception as e:
        print(f"  '{q[:50]}': FAIL {e}")
gate=EpisodeGate(GateCfg(min_sources=1,min_consistency=0.0,uncertainty_thresh=0.0))
gr=gate.check(all_eps)
print(f"gate: accept={gr.accept} n={gr.n_sources} cons={gr.consistency:.3f}")
for e in all_eps[:8]:
    print(f"  [{e.eid[:8]}] auth={e.authority:.3f} '{e.title[:70]}'")
store=EpisodeStore("/tmp/wm_scale_ep.db")
store.put_many(all_eps)
print(f"store: {store.count()} eps ({ts()-t0:.1f}s)")

pr("PHASE 2: CHUNK + MULTI-FORMAT DATASET")
chunks=Chunker(ChunkCfg(max_tok=256,overlap=32)).chunk_many(store.all())
ds_cpt=DatasetBuilder(DatasetCfg(recipe="cpt")).build(chunks)
ds_sft=DatasetBuilder(DatasetCfg(recipe="ext_sft")).build(chunks)
ds_qa=DatasetBuilder(DatasetCfg(recipe="cited_qa")).build(chunks)
print(f"chunks={len(chunks)} cpt={len(ds_cpt)} sft={len(ds_sft)} qa={len(ds_qa)}")
pb=ProbeBuilder()
all_probes=pb.build_all(chunks)
print(f"probes: {len(all_probes)} (cloze={sum(1 for p in all_probes if p.kind=='cloze')}, qa={sum(1 for p in all_probes if p.kind=='qa')})")
ds_eval=sub_ds(ds_cpt)
ds_eval_sft=sub_ds(ds_sft)
print(f"eval: cpt={len(ds_eval)} sft={len(ds_eval_sft)}")

pr("PHASE 3: LOAD Qwen2.5-1.5B")
t0=ts()
mn="Qwen/Qwen2.5-1.5B"
tok=AutoTokenizer.from_pretrained(mn,trust_remote_code=True)
if tok.pad_token is None:tok.pad_token=tok.eos_token
base=AutoModelForCausalLM.from_pretrained(mn,torch_dtype=torch.float32,trust_remote_code=True)
lc=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,
    target_modules=["q_proj","v_proj"],task_type=TaskType.CAUSAL_LM)
base=get_peft_model(base,lc)
base.print_trainable_parameters()
snap={k:v.clone() for k,v in base.state_dict().items()}
print(f"load: {ts()-t0:.1f}s")

print("baseline eval...")
baseline=eval_full(base,tok,ds_eval,all_probes,"baseline")
print(f"baseline done ({ts()-t0:.1f}s)")
results={"baseline":baseline}

pr(f"PHASE 4: DPMU at scale ({STEPS} steps, lr={LR}, r=16, ml={ML})")
t0=ts()
m_dpmu=copy.deepcopy(base);m_dpmu.load_state_dict(snap)
t_dpmu=copy.deepcopy(m_dpmu).eval()
r_dpmu=DPMURunner(lr=LR,max_steps=STEPS,bs=BS,temp=2.0,
    n_dream_grads=3,max_len=ML,dream_n=DN,dream_len=DL)
tr_dpmu=r_dpmu.run(m_dpmu,t_dpmu,ds_cpt,P_DREAM,tok)
print(f"  train: loss={tr_dpmu.loss:.6f} dream={tr_dpmu.dream_loss:.6f} steps={tr_dpmu.steps}")
del t_dpmu;gc.collect()
print("  eval...")
dpmu_m=eval_full(m_dpmu,tok,ds_eval,all_probes,"DPMU-scale")
dpmu_m["train_loss"]=tr_dpmu.loss
dpmu_m["dream_loss"]=tr_dpmu.dream_loss
dpmu_m["time"]=ts()-t0
dk=drift_kl(m_dpmu,base,tok,ANCHORS,max_len=128)
dpmu_m["drift_kl"]=dk
results["dpmu_scale"]=dpmu_m
print(f"  drift_kl={dk:.8f} time={dpmu_m['time']:.1f}s")

pr(f"PHASE 5: EATRD→DPMU→SSC at scale ({STEPS} total steps)")
t0=ts()
m_pipe=copy.deepcopy(base);m_pipe.load_state_dict(snap)

eatrd_steps=STEPS*2//5
dpmu_steps=STEPS*2//5
ssc_refine=STEPS//5
print(f"  split: EATRD={eatrd_steps}st, DPMU={dpmu_steps}st, SSC-refine={ssc_refine}st")

print(f"  [1/3] EATRD phase ({eatrd_steps} steps)...")
t_e=copy.deepcopy(m_pipe).eval()
r_e=EATRDRunner(lr=LR,max_steps=eatrd_steps,bs=BS,temp=2.0,
    eps_min=0.01,alpha=0.5,rho=0.01,lam_init=1.0,
    max_len=ML,dream_n=DN,dream_len=DL)
tr_e=r_e.run(m_pipe,t_e,ds_cpt,P_DREAM,tok)
print(f"    loss={tr_e.loss:.6f} dream={tr_e.dream_loss:.6f} lam={tr_e.extras.get('lambda',0):.4f}")
del t_e;gc.collect()

print(f"  [2/3] DPMU phase ({dpmu_steps} steps)...")
t_d=copy.deepcopy(m_pipe).eval()
r_d=DPMURunner(lr=LR*0.7,max_steps=dpmu_steps,bs=BS,temp=2.0,
    n_dream_grads=3,max_len=ML,dream_n=DN,dream_len=DL)
tr_d=r_d.run(m_pipe,t_d,ds_cpt,P_DREAM,tok)
print(f"    loss={tr_d.loss:.6f} dream={tr_d.dream_loss:.6f}")
del t_d;gc.collect()

print(f"  [3/3] SSC consolidation (refine={ssc_refine} steps)...")
bank=AdapterBank();bank.add(m_pipe,weight=1.0)
sc=SleepConsolidator(rank=16,beta=0.1,gamma=0.01,tau=1.0,
    refine_steps=ssc_refine,lr=1e-4,temp=2.0,dream_n=DN,dream_len=DL)
sc.consolidate(bank,m_pipe,P_DREAM,tok)

print("  eval...")
pipe_m=eval_full(m_pipe,tok,ds_eval,all_probes,"EATRD→DPMU→SSC-scale")
pipe_m["train_loss"]=(tr_e.loss+tr_d.loss)/2
pipe_m["dream_loss"]=(tr_e.dream_loss+tr_d.dream_loss)/2
pipe_m["time"]=ts()-t0
dk=drift_kl(m_pipe,base,tok,ANCHORS,max_len=128)
pipe_m["drift_kl"]=dk
results["eatrd_dpmu_ssc_scale"]=pipe_m
print(f"  drift_kl={dk:.8f} time={pipe_m['time']:.1f}s")

pr("PHASE 6: UPDATEGUARD VERIFICATION")
for rname,label in [("dpmu_scale","DPMU"),("eatrd_dpmu_ssc_scale","EATRD→DPMU→SSC")]:
    m=results[rname]
    pre_acc=[baseline["acc"]]*20
    post_acc=[m["acc"]]*20
    pre_anc=baseline["per_anchor"]
    post_anc=m["per_anchor"]
    ci_lo,ci_hi=paired_bootstrap_ci(pre_acc,post_acc,n_boot=1000,alpha=0.05)
    sv=sprt_test(pre_acc,post_acc,delta=0.005)
    anc_lo,anc_hi=paired_bootstrap_ci(pre_anc,post_anc,n_boot=1000,alpha=0.05)
    print(f"  {label}:")
    print(f"    acc CI=[{ci_lo:+.6f}, {ci_hi:+.6f}] SPRT={sv}")
    print(f"    anchor CI=[{anc_lo:+.6f}, {anc_hi:+.6f}]")
    m["acc_ci"]=(ci_lo,ci_hi)
    m["anc_ci"]=(anc_lo,anc_hi)
    m["sprt"]=sv

pr("PHASE 7: PAIRED CERTIFICATION")
cert=Certifier(CertCfg(n_bootstrap=1000,ci_alpha=0.05,sprt_delta=0.005,
    sprt_alpha=0.05,sprt_beta=0.1,max_drift=0.3))
for rname,label in [("dpmu_scale","DPMU"),("eatrd_dpmu_ssc_scale","EATRD→DPMU→SSC")]:
    m=results[rname]
    pre_s=[baseline["acc"]]*20
    post_s=[m["acc"]]*20
    pre_a=baseline["per_anchor"]
    post_a=m["per_anchor"]
    cr=cert.certify_paired(pre_s,post_s,pre_a,post_a)
    m["certified"]=cr.passed
    print(f"  {label}: CERTIFIED={cr.passed}")
    for c in cr.checks:
        print(f"    {c.name}: passed={c.passed} val={c.val:.6f} thresh={c.thresh:.6f}")

pr("PHASE 8: HEAD-TO-HEAD COMPARISON")
print()
hdr=f"{'Metric':<20}|{'Baseline':>12}|{'DPMU':>12}|{'E→D→S':>12}"
print(hdr);print("-"*len(hdr))
for metric,key,fmt in [
    ("PPL","ppl",".2f"),("Accuracy","acc",".4f"),
    ("Probe Hit Rate","hit_rate",".4f"),("Probe F1","mean_f1",".4f"),
    ("Anchor NLL","anchor_nll",".4f"),("Drift KL","drift_kl",".8f"),
    ("Train Loss","train_loss",".6f"),("Dream Loss","dream_loss",".6f"),
    ("Time (s)","time",".0f"),
]:
    bv=baseline.get(key,0)
    dv=results["dpmu_scale"].get(key,0)
    pv=results["eatrd_dpmu_ssc_scale"].get(key,0)
    print(f"{metric:<20}|{bv:>12{fmt}}|{dv:>12{fmt}}|{pv:>12{fmt}}")

print()
d_da=results["dpmu_scale"]["acc"]-baseline["acc"]
p_da=results["eatrd_dpmu_ssc_scale"]["acc"]-baseline["acc"]
d_dp=results["dpmu_scale"]["ppl"]-baseline["ppl"]
p_dp=results["eatrd_dpmu_ssc_scale"]["ppl"]-baseline["ppl"]
d_dn=results["dpmu_scale"]["anchor_nll"]-baseline["anchor_nll"]
p_dn=results["eatrd_dpmu_ssc_scale"]["anchor_nll"]-baseline["anchor_nll"]
print(f"DPMU delta:      acc={d_da:+.4f}  ppl={d_dp:+.2f}  anchor={d_dn:+.4f}  drift={results['dpmu_scale']['drift_kl']:.8f}")
print(f"E→D→S delta:     acc={p_da:+.4f}  ppl={p_dp:+.2f}  anchor={p_dn:+.4f}  drift={results['eatrd_dpmu_ssc_scale']['drift_kl']:.8f}")

if d_da>p_da and results["dpmu_scale"]["drift_kl"]<=results["eatrd_dpmu_ssc_scale"]["drift_kl"]:
    winner="DPMU"
elif p_da>d_da and p_dn<d_dn:
    winner="EATRD→DPMU→SSC"
elif d_da>p_da:
    winner="DPMU (higher acc)"
else:
    winner="EATRD→DPMU→SSC (better anchor stability)"
print(f"\nWINNER: {winner}")

out={}
for k,v in results.items():
    out[k]={kk:(list(vv) if isinstance(vv,tuple) else vv) for kk,vv in v.items() if kk!="per_anchor"}
with open("/tmp/wm_scale_results.json","w") as f:
    json.dump({"results":out,"winner":winner},f,indent=2,default=str)
print(f"Saved to /tmp/wm_scale_results.json")
store.close()
pr("SCALE BENCHMARK COMPLETE")
