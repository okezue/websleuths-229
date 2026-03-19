#!/usr/bin/env python3
import os,sys,copy,time,json,torch,gc,random
os.environ["EXA_API_KEY"]=os.environ.get("EXA_API_KEY","")
os.environ["TOKENIZERS_PARALLELISM"]="false"
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import get_peft_model,LoraConfig,TaskType
from datasets import Dataset
from wm.ingest.exa import ExaSrc
from wm.store import EpisodeStore
from wm.chunk import Chunker
from wm.dataset import DatasetBuilder
from wm.gate import EpisodeGate,exa_authority,SearchGate,UpdateGate
from wm.recipe import EATRDRunner,DPMURunner,EABSSCRunner,AdapterBank,SleepConsolidator
from wm.eval import Evaluator,ProbeBuilder,AnchorEval
from wm.guard import UpdateGuard,paired_bootstrap_ci,sprt_test
from wm.guard.drift import drift_kl
from wm.cert import Certifier
from wm.search.agent import AgenticSearcher
from wm.search.claims import extract_claims,extract_entities
from wm.graph.store import GraphStore
from wm.graph.community import detect_communities
from wm.graph.dream_gen import gen_dream_prompts
from wm.pace.controller import PaceController
from wm.pipe.loop import AgenticPipeline
from wm.cfg import *
from wm.types import Chunk

DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {DEV}")
if DEV.type=="cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f}GB")

ANCHORS=[
    "The Earth revolves around the Sun.",
    "Water freezes at zero degrees Celsius.",
    "The speed of light is approximately 300000 kilometers per second.",
    "DNA carries genetic information in living organisms.",
    "Gravity pulls objects toward the center of the Earth.",
    "The chemical formula for water is H2O.",
    "Photosynthesis converts sunlight into chemical energy.",
    "Oxygen is essential for human respiration.",
    "The Pacific Ocean is the largest ocean on Earth.",
    "Iron is a magnetic metal.",
]
STEPS=200
BS=2
LR=3e-4
ML=256
DN=4
DL=64
EVAL_N=50
PROBE_N=30
MN="Qwen/Qwen2.5-1.5B"

def pr(msg):print(f"\n{'='*70}\n{msg}\n{'='*70}",flush=True)
def ts():return time.time()
def sub_ds(ds,n=EVAL_N):
    if len(ds)<=n:return ds
    idx=list(range(len(ds)));random.seed(42);random.shuffle(idx)
    return ds.select(idx[:n])

def eval_full(model,tok,ds_eval,probes,tag=""):
    model.to(DEV)
    ev=Evaluator(model,tok,max_len=ML)
    ev._dev=DEV
    report=ev.evaluate(ds_eval)
    pscore=ev.evaluate_probes(probes[:PROBE_N])
    anc=AnchorEval(model,tok,ANCHORS)
    anll=anc.nll()
    per_anc=anc.per_anchor_nll()
    print(f"  [{tag}] ppl={report.ppl:.2f} acc={report.acc:.4f}")
    print(f"  [{tag}] probes: hit={pscore['hit_rate']:.4f} f1={pscore['mean_f1']:.4f} n={pscore['n_probes']}")
    print(f"  [{tag}] anchor_nll={anll:.4f}",flush=True)
    return {"ppl":report.ppl,"acc":report.acc,"hit_rate":pscore["hit_rate"],
            "mean_f1":pscore["mean_f1"],"anchor_nll":anll,
            "per_anchor":per_anc,"n_probes":pscore["n_probes"]}

results={}

pr(f"PHASE 1: LOAD {MN} + LoRA r=32")
t0=ts()
tok=AutoTokenizer.from_pretrained(MN,trust_remote_code=True)
if tok.pad_token is None:tok.pad_token=tok.eos_token
base=AutoModelForCausalLM.from_pretrained(MN,torch_dtype=torch.bfloat16,trust_remote_code=True)
base.to(DEV)
lc=LoraConfig(r=32,lora_alpha=64,lora_dropout=0.05,
    target_modules=["q_proj","v_proj"],task_type=TaskType.CAUSAL_LM)
base=get_peft_model(base,lc)
base.print_trainable_parameters()
snap={k:v.clone() for k,v in base.state_dict().items()}
print(f"load: {ts()-t0:.1f}s",flush=True)

pr("PHASE 2: AGENTIC SEARCH — multi-domain ingestion")
t0=ts()
scfg=SearchCfg(max_rounds=3,queries_per_round=5,min_claims=10,
    mmr_lambda=0.7,mmr_k=40,exa_api_key=os.environ["EXA_API_KEY"],
    model_query_gen=True,query_temp=0.9)
gcfg=GraphCfg(db_path="/tmp/wm_agentic_graph.db")
gs=GraphStore(gcfg.db_path)

DOMAINS={
    "forensic_science":[
        "unsolved cold case forensic evidence DNA 2024",
        "genetic genealogy criminal identification breakthroughs",
        "forensic pathology autopsy techniques modern",
        "digital forensics cybercrime evidence recovery",
        "forensic ballistics gunshot residue analysis",
    ],
    "finance":[
        "financial ratio analysis earnings reports SEC filings 2024",
        "quantitative trading strategies alpha generation",
        "credit risk modeling Basel IV regulatory framework",
        "ESG investing metrics sustainability reporting",
    ],
    "legal":[
        "legal precedent constitutional law Supreme Court 2024",
        "criminal procedure Fourth Amendment digital search",
        "international humanitarian law armed conflict",
        "intellectual property AI generated content copyright",
    ],
    "chemistry":[
        "organic chemistry reaction mechanisms catalysis 2024",
        "CRISPR gene editing delivery mechanisms",
        "computational chemistry molecular dynamics simulation",
        "green chemistry sustainable synthesis methods",
    ],
}
all_eps=[]
src=ExaSrc()
domain_eps={}
for dom,queries in DOMAINS.items():
    print(f"\n  [{dom}] searching...",flush=True)
    dom_eps=[]
    for q in queries:
        try:
            eps=src.fetch(q,n=8)
            eps=exa_authority(eps,q)
            dom_eps.extend(eps)
            print(f"    '{q[:60]}': {len(eps)} eps")
        except Exception as e:
            print(f"    '{q[:60]}': FAIL {e}")
    all_eps.extend(dom_eps)
    domain_eps[dom]=dom_eps
    print(f"  [{dom}] total: {len(dom_eps)} eps",flush=True)

store=EpisodeStore("/tmp/wm_agentic_ep.db")
store.put_many(all_eps)
print(f"\nstore: {store.count()} total eps ({ts()-t0:.1f}s)",flush=True)

pr("PHASE 3: AGENTIC CLAIM EXTRACTION + EEG CONSTRUCTION")
t0=ts()
all_chunks_by_dom={}
all_claims_by_dom={}
all_communities_by_dom={}
domain_dreams={}
chunker=Chunker(ChunkCfg(max_tok=256,overlap=32))

for dom,deps in domain_eps.items():
    print(f"\n  [{dom}] chunking + claim extraction...",flush=True)
    chunks=chunker.chunk_many(deps)
    claims=[]
    for ci,c in enumerate(chunks):
        cls=extract_claims(c.text,eid=c.eid,chunk_idx=ci)
        claims.extend(cls)
    ents=extract_entities(claims)
    comms=detect_communities(claims,thresh=0.5) if claims else []
    all_chunks_by_dom[dom]=chunks
    all_claims_by_dom[dom]=claims
    all_communities_by_dom[dom]=comms
    for c in claims:gs.put_claim(c)
    for e in ents:gs.put_entity(e)
    for co in comms:gs.put_community(co)
    dreams=gen_dream_prompts(comms,claims,n_general=5)
    domain_dreams[dom]=dreams
    print(f"  [{dom}] chunks={len(chunks)} claims={len(claims)} entities={len(ents)} "
          f"communities={len(comms)} dreams={len(dreams)}",flush=True)

total_claims=sum(len(v) for v in all_claims_by_dom.values())
total_comms=sum(len(v) for v in all_communities_by_dom.values())
print(f"\nEEG totals: claims={total_claims} communities={total_comms} ({ts()-t0:.1f}s)",flush=True)

pr("PHASE 4: DATASET CONSTRUCTION (all domains)")
all_chunks=[]
for dom in DOMAINS:all_chunks.extend(all_chunks_by_dom.get(dom,[]))
ds_cpt=DatasetBuilder(DatasetCfg(recipe="cpt")).build(all_chunks)
ds_sft=DatasetBuilder(DatasetCfg(recipe="ext_sft")).build(all_chunks)
ds_qa=DatasetBuilder(DatasetCfg(recipe="cited_qa")).build(all_chunks)
print(f"datasets: cpt={len(ds_cpt)} sft={len(ds_sft)} qa={len(ds_qa)}")
pb=ProbeBuilder()
all_probes=pb.build_all(all_chunks)
print(f"probes: {len(all_probes)} (cloze={sum(1 for p in all_probes if p.kind=='cloze')}, "
      f"qa={sum(1 for p in all_probes if p.kind=='qa')})")
ds_eval=sub_ds(ds_cpt)
all_dreams=[]
for dom in DOMAINS:all_dreams.extend(domain_dreams.get(dom,[]))
print(f"domain dreams: {len(all_dreams)} (vs generic trivia)")

pr("PHASE 5: SEARCH GATE + UPDATE GATE VALIDATION")
sgcfg=SearchGateCfg(a=0.4,b=0.3,c=0.3,tau_search=0.5)
sgat=SearchGate(sgcfg)
ugcfg=UpdateGateCfg(a1=0.3,a2=0.2,a3=0.3,a4=0.2,tau_ready=0.4,tau_novel=0.2)
ugat=UpdateGate(ugcfg)
from wm.types import SearchResult
for dom in DOMAINS:
    claims=all_claims_by_dom.get(dom,[])
    comms=all_communities_by_dom.get(dom,[])
    srcs=[e.url for e in domain_eps.get(dom,[])]
    sr=SearchResult(topic=dom,claims=claims,communities=comms,
                    chunks=all_chunks_by_dom.get(dom,[]),sources=srcs)
    sgr=sgat.check(dom,communities=comms,model=base,tok=tok)
    ugr=ugat.check(sr,model=base,tok=tok)
    print(f"  [{dom}] search_gate: search={sgr.should_search} score={sgr.score:.3f} "
          f"unc={sgr.uncertainty:.3f} fresh={sgr.freshness:.1f} param={sgr.param_match:.3f}")
    print(f"  [{dom}] update_gate: update={ugr.should_update} ready={ugr.readiness:.3f} "
          f"novel={ugr.novelty:.3f}",flush=True)

pr("PHASE 6: BASELINE EVAL")
t0=ts()
baseline=eval_full(base,tok,ds_eval,all_probes,"baseline")
results["baseline"]=baseline
print(f"baseline done ({ts()-t0:.1f}s)",flush=True)

pr(f"PHASE 7: DPMU with DOMAIN DREAMS ({STEPS} steps, lr={LR})")
t0=ts()
m_dpmu=copy.deepcopy(base);m_dpmu.load_state_dict(snap);m_dpmu.to(DEV)
t_dpmu=copy.deepcopy(m_dpmu).eval()
r_dpmu=DPMURunner(lr=LR,max_steps=STEPS,bs=BS,temp=2.0,
    n_dream_grads=3,max_len=ML,dream_n=DN,dream_len=DL)
tr_dpmu=r_dpmu.run(m_dpmu,t_dpmu,ds_cpt,all_dreams,tok)
print(f"  train: loss={tr_dpmu.loss:.6f} dream={tr_dpmu.dream_loss:.6f} steps={tr_dpmu.steps}")
del t_dpmu;gc.collect();torch.cuda.empty_cache() if DEV.type=="cuda" else None
dpmu_m=eval_full(m_dpmu,tok,ds_eval,all_probes,"DPMU-domain-dreams")
dpmu_m["train_loss"]=tr_dpmu.loss
dpmu_m["dream_loss"]=tr_dpmu.dream_loss
dpmu_m["time"]=ts()-t0
dk=drift_kl(m_dpmu,base,tok,ANCHORS,max_len=128)
dpmu_m["drift_kl"]=dk
results["dpmu_domain"]=dpmu_m
print(f"  drift_kl={dk:.8f} time={dpmu_m['time']:.1f}s",flush=True)

pr(f"PHASE 8: DPMU with GENERIC DREAMS (control — {STEPS} steps)")
t0=ts()
P_GENERIC=[
    "What is the capital of France?","Explain photosynthesis briefly.",
    "Who wrote Hamlet?","What causes earthquakes?",
    "How does the internet work?","What is DNA?",
    "Describe the water cycle.","What is machine learning?",
    "Who was Albert Einstein?","How do airplanes fly?",
    "What is the speed of light?","How do vaccines work?",
    "What causes the seasons?","What is evolution?",
    "How does electricity work?","What is the periodic table?",
]
m_gen=copy.deepcopy(base);m_gen.load_state_dict(snap);m_gen.to(DEV)
t_gen=copy.deepcopy(m_gen).eval()
r_gen=DPMURunner(lr=LR,max_steps=STEPS,bs=BS,temp=2.0,
    n_dream_grads=3,max_len=ML,dream_n=DN,dream_len=DL)
tr_gen=r_gen.run(m_gen,t_gen,ds_cpt,P_GENERIC,tok)
print(f"  train: loss={tr_gen.loss:.6f} dream={tr_gen.dream_loss:.6f} steps={tr_gen.steps}")
del t_gen;gc.collect();torch.cuda.empty_cache() if DEV.type=="cuda" else None
gen_m=eval_full(m_gen,tok,ds_eval,all_probes,"DPMU-generic-dreams")
gen_m["train_loss"]=tr_gen.loss
gen_m["dream_loss"]=tr_gen.dream_loss
gen_m["time"]=ts()-t0
dk2=drift_kl(m_gen,base,tok,ANCHORS,max_len=128)
gen_m["drift_kl"]=dk2
results["dpmu_generic"]=gen_m
print(f"  drift_kl={dk2:.8f} time={gen_m['time']:.1f}s",flush=True)

pr(f"PHASE 9: EATRD with DOMAIN DREAMS ({STEPS} steps)")
t0=ts()
m_eatrd=copy.deepcopy(base);m_eatrd.load_state_dict(snap);m_eatrd.to(DEV)
t_eatrd=copy.deepcopy(m_eatrd).eval()
r_eatrd=EATRDRunner(lr=LR,max_steps=STEPS,bs=BS,temp=2.0,
    eps_min=0.01,alpha=0.5,rho=0.01,lam_init=1.0,
    max_len=ML,dream_n=DN,dream_len=DL)
tr_eatrd=r_eatrd.run(m_eatrd,t_eatrd,ds_cpt,all_dreams,tok)
print(f"  train: loss={tr_eatrd.loss:.6f} dream={tr_eatrd.dream_loss:.6f} "
      f"lam={tr_eatrd.extras.get('lambda',0):.4f} eps_k={tr_eatrd.extras.get('eps_k',0):.4f}")
del t_eatrd;gc.collect();torch.cuda.empty_cache() if DEV.type=="cuda" else None
eatrd_m=eval_full(m_eatrd,tok,ds_eval,all_probes,"EATRD-domain")
eatrd_m["train_loss"]=tr_eatrd.loss
eatrd_m["dream_loss"]=tr_eatrd.dream_loss
eatrd_m["time"]=ts()-t0
dk3=drift_kl(m_eatrd,base,tok,ANCHORS,max_len=128)
eatrd_m["drift_kl"]=dk3
results["eatrd_domain"]=eatrd_m
print(f"  drift_kl={dk3:.8f} time={eatrd_m['time']:.1f}s",flush=True)

pr(f"PHASE 10: EAB-SSC with DOMAIN DREAMS ({STEPS} steps)")
t0=ts()
m_eab=copy.deepcopy(base);m_eab.load_state_dict(snap);m_eab.to(DEV)
t_eab=copy.deepcopy(m_eab).eval()
r_eab=EABSSCRunner(lr=LR,max_steps=STEPS,bs=BS,temp=2.0,
    dream_weight=0.5,max_len=ML,dream_n=DN,dream_len=DL)
tr_eab=r_eab.day(m_eab,t_eab,ds_cpt,all_dreams,tok)
print(f"  train: loss={tr_eab.loss:.6f} dream={tr_eab.dream_loss:.6f}")
del t_eab;gc.collect();torch.cuda.empty_cache() if DEV.type=="cuda" else None
eab_m=eval_full(m_eab,tok,ds_eval,all_probes,"EAB-SSC-domain")
eab_m["train_loss"]=tr_eab.loss
eab_m["dream_loss"]=tr_eab.dream_loss
eab_m["time"]=ts()-t0
dk4=drift_kl(m_eab,base,tok,ANCHORS,max_len=128)
eab_m["drift_kl"]=dk4
results["eab_ssc_domain"]=eab_m
print(f"  drift_kl={dk4:.8f} time={eab_m['time']:.1f}s",flush=True)

pr(f"PHASE 11: FULL PIPELINE — EATRD→DPMU→SSC + domain dreams ({STEPS} total)")
t0=ts()
m_pipe=copy.deepcopy(base);m_pipe.load_state_dict(snap);m_pipe.to(DEV)
eatrd_steps=STEPS*2//5
dpmu_steps=STEPS*2//5
ssc_refine=STEPS//5
print(f"  split: EATRD={eatrd_steps}st DPMU={dpmu_steps}st SSC-refine={ssc_refine}st")

print(f"  [1/3] EATRD ({eatrd_steps} steps)...",flush=True)
te1=copy.deepcopy(m_pipe).eval()
re1=EATRDRunner(lr=LR,max_steps=eatrd_steps,bs=BS,temp=2.0,
    eps_min=0.01,alpha=0.5,rho=0.01,lam_init=1.0,
    max_len=ML,dream_n=DN,dream_len=DL)
tre1=re1.run(m_pipe,te1,ds_cpt,all_dreams,tok)
print(f"    loss={tre1.loss:.6f} dream={tre1.dream_loss:.6f} lam={tre1.extras.get('lambda',0):.4f}")
del te1;gc.collect();torch.cuda.empty_cache() if DEV.type=="cuda" else None

print(f"  [2/3] DPMU ({dpmu_steps} steps)...",flush=True)
td1=copy.deepcopy(m_pipe).eval()
rd1=DPMURunner(lr=LR*0.7,max_steps=dpmu_steps,bs=BS,temp=2.0,
    n_dream_grads=3,max_len=ML,dream_n=DN,dream_len=DL)
trd1=rd1.run(m_pipe,td1,ds_cpt,all_dreams,tok)
print(f"    loss={trd1.loss:.6f} dream={trd1.dream_loss:.6f}")
del td1;gc.collect();torch.cuda.empty_cache() if DEV.type=="cuda" else None

print(f"  [3/3] SSC consolidation (refine={ssc_refine})...",flush=True)
bank=AdapterBank();bank.add(m_pipe,weight=1.0)
sc=SleepConsolidator(rank=32,beta=0.1,gamma=0.01,tau=1.0,
    refine_steps=ssc_refine,lr=1e-4,temp=2.0,dream_n=DN,dream_len=DL)
sc.consolidate(bank,m_pipe,all_dreams,tok)

pipe_m=eval_full(m_pipe,tok,ds_eval,all_probes,"EATRD→DPMU→SSC-domain")
pipe_m["train_loss"]=(tre1.loss+trd1.loss)/2
pipe_m["dream_loss"]=(tre1.dream_loss+trd1.dream_loss)/2
pipe_m["time"]=ts()-t0
dk5=drift_kl(m_pipe,base,tok,ANCHORS,max_len=128)
pipe_m["drift_kl"]=dk5
results["eatrd_dpmu_ssc_domain"]=pipe_m
print(f"  drift_kl={dk5:.8f} time={pipe_m['time']:.1f}s",flush=True)

pr("PHASE 12: DOMAIN RETENTION TEST")
t0=ts()
domain_probes={}
for dom in DOMAINS:
    dchunks=all_chunks_by_dom.get(dom,[])
    if dchunks:
        dp=pb.build_all(dchunks)
        domain_probes[dom]=dp[:PROBE_N]
        print(f"  [{dom}] probes={len(dp)}",flush=True)

retention={}
for rname in ["dpmu_domain","eatrd_domain","eab_ssc_domain","eatrd_dpmu_ssc_domain"]:
    model_ref={"dpmu_domain":m_dpmu,"eatrd_domain":m_eatrd,
               "eab_ssc_domain":m_eab,"eatrd_dpmu_ssc_domain":m_pipe}[rname]
    ev=Evaluator(model_ref,tok,max_len=ML)
    ev._dev=DEV
    ret={}
    for dom,dprobes in domain_probes.items():
        sc=ev.evaluate_probes(dprobes)
        ret[dom]=sc
    retention[rname]=ret
    print(f"  {rname}:")
    for dom,sc in ret.items():
        print(f"    {dom}: hit={sc['hit_rate']:.4f} f1={sc['mean_f1']:.4f}")
results["domain_retention"]=retention
print(f"retention done ({ts()-t0:.1f}s)",flush=True)

pr("PHASE 13: PACE CONTROLLER DEMO")
pcfg=PaceCfg(web_budget=len(DOMAINS)*5,ft_budget=5,eta=0.1)
pc=PaceController(pcfg)
for dom in DOMAINS:
    pc.on_search()
    print(f"  after {dom} search: tau_search={pc.tau_search:.3f} lam_web={pc.state.lam_web:.3f}")
pc.on_ft()
print(f"  after FT: tau_ready={pc.tau_ready:.3f} lam_ft={pc.state.lam_ft:.3f}")

pr("PHASE 14: UPDATEGUARD + CERTIFICATION")
for rname,label in [("dpmu_domain","DPMU-domain"),("dpmu_generic","DPMU-generic"),
                    ("eatrd_domain","EATRD-domain"),("eab_ssc_domain","EAB-SSC-domain"),
                    ("eatrd_dpmu_ssc_domain","E→D→S-domain")]:
    m=results[rname]
    pre_acc=[baseline["acc"]]*20
    post_acc=[m["acc"]]*20
    pre_anc=baseline["per_anchor"]
    post_anc=m["per_anchor"]
    ci_lo,ci_hi=paired_bootstrap_ci(pre_acc,post_acc,n_boot=1000,alpha=0.05)
    sv=sprt_test(pre_acc,post_acc,delta=0.005)
    anc_lo,anc_hi=paired_bootstrap_ci(pre_anc,post_anc,n_boot=1000,alpha=0.05)
    cert=Certifier(CertCfg(n_bootstrap=1000,ci_alpha=0.05,sprt_delta=0.005,
        sprt_alpha=0.05,sprt_beta=0.1,max_drift=0.3))
    cr=cert.certify_paired(pre_acc,post_acc,pre_anc,post_anc)
    m["certified"]=cr.passed
    m["acc_ci"]=(ci_lo,ci_hi)
    m["anc_ci"]=(anc_lo,anc_hi)
    m["sprt"]=sv
    print(f"  {label}: CERT={cr.passed} acc_CI=[{ci_lo:+.6f},{ci_hi:+.6f}] SPRT={sv}")
    for c in cr.checks:
        print(f"    {c.name}: p={c.passed} v={c.val:.6f}",flush=True)

pr("PHASE 15: DOMAIN vs GENERIC DREAM COMPARISON")
dd=results["dpmu_domain"]
dg=results["dpmu_generic"]
print(f"{'Metric':<20}|{'Domain Dreams':>14}|{'Generic Dreams':>14}|{'Delta':>10}")
print("-"*62)
for metric,key,fmt in [
    ("PPL","ppl",".2f"),("Accuracy","acc",".4f"),
    ("Probe Hit","hit_rate",".4f"),("Probe F1","mean_f1",".4f"),
    ("Anchor NLL","anchor_nll",".4f"),("Drift KL","drift_kl",".8f"),
    ("Dream Loss","dream_loss",".6f"),("Time (s)","time",".0f"),
]:
    dv=dd.get(key,0);gv=dg.get(key,0);delta=dv-gv
    print(f"{metric:<20}|{dv:>14{fmt}}|{gv:>14{fmt}}|{delta:>+10{fmt}}")
if dd["mean_f1"]>dg["mean_f1"]:
    print("\nDomain dreams produce BETTER probe retention (fixes DPMU zero-out)")
elif dd["mean_f1"]==dg["mean_f1"]:
    print("\nTied on probe F1")
else:
    print("\nGeneric dreams won on probe F1")

pr("PHASE 16: FULL RESULTS TABLE")
print()
hdr=f"{'Recipe':<26}|{'PPL':>7}|{'Acc':>7}|{'Hit':>6}|{'F1':>6}|{'AncNLL':>7}|{'Drift':>10}|{'DreamL':>7}|{'Time':>6}|{'Cert':>5}"
print(hdr);print("-"*len(hdr))
for name,m in results.items():
    if name in ("baseline","domain_retention"):
        if name=="baseline":
            print(f"{'baseline':<26}|{m.get('ppl',0):>7.2f}|{m.get('acc',0):>7.4f}|{m.get('hit_rate',0):>6.3f}|{m.get('mean_f1',0):>6.3f}|{m.get('anchor_nll',0):>7.3f}|{'---':>10}|{'---':>7}|{'---':>6}|{'---':>5}")
        continue
    ct="Y" if m.get("certified") else "N"
    print(f"{name:<26}|{m.get('ppl',0):>7.2f}|{m.get('acc',0):>7.4f}|{m.get('hit_rate',0):>6.3f}|{m.get('mean_f1',0):>6.3f}|{m.get('anchor_nll',0):>7.3f}|{m.get('drift_kl',0):>10.7f}|{m.get('dream_loss',0):>7.4f}|{m.get('time',0):>5.0f}s|{ct:>5}")

pr("PHASE 17: RANKINGS")
recipe_results=[(k,v) for k,v in results.items() if k not in ("baseline","domain_retention")]
ranked_f1=sorted(recipe_results,key=lambda x:(-x[1].get("mean_f1",0),x[1].get("drift_kl",999)))
print("By probe F1 (tiebreak: drift):")
for i,(n,m) in enumerate(ranked_f1):
    ct=" CERT" if m.get("certified") else ""
    print(f"  #{i+1} {n}: f1={m['mean_f1']:.4f} drift={m.get('drift_kl',0):.7f}{ct}")

ranked_ppl=sorted(recipe_results,key=lambda x:x[1]["ppl"])
print("\nBy perplexity:")
for i,(n,m) in enumerate(ranked_ppl):
    print(f"  #{i+1} {n}: ppl={m['ppl']:.2f} acc={m['acc']:.4f}")

ranked_drift=sorted(recipe_results,key=lambda x:x[1].get("drift_kl",999))
print("\nBy drift (stability):")
for i,(n,m) in enumerate(ranked_drift):
    print(f"  #{i+1} {n}: drift={m.get('drift_kl',0):.7f} anc={m['anchor_nll']:.4f}")

bl_ppl=max(baseline["ppl"],1);bl_anc=baseline["anchor_nll"]
composite=sorted(recipe_results,
    key=lambda x:(-x[1].get("acc",0)*0.25-x[1].get("mean_f1",0)*0.25
                  +x[1].get("drift_kl",0)*100*0.25+(x[1]["anchor_nll"]-bl_anc)*0.25))
print("\nComposite (25% acc, 25% f1, 25% -drift, 25% -anchor_delta):")
for i,(n,m) in enumerate(composite):
    sc=(m.get("acc",0)*0.25+m.get("mean_f1",0)*0.25
        -m.get("drift_kl",0)*100*0.25-(m["anchor_nll"]-bl_anc)*0.25)
    print(f"  #{i+1} {n}: composite={sc:.4f}")

winner=composite[0][0]
print(f"\nOVERALL WINNER: {winner}")

pr("PHASE 18: EEG SUMMARY")
print(f"Graph store: {gs.count_claims()} claims")
comms=gs.get_communities()
print(f"  communities: {len(comms)}")
for co in comms[:10]:
    print(f"    [{co.coid[:8]}] {co.label}: {len(co.members)} members, param={co.parameterized}")
ents=gs.get_entities()
print(f"  entities: {len(ents)}")
for e in ents[:15]:
    print(f"    {e.name}")

pr("SAVE RESULTS")
out={}
for k,v in results.items():
    if k=="domain_retention":
        out[k]=v
        continue
    out[k]={kk:(list(vv) if isinstance(vv,tuple) else vv) for kk,vv in v.items() if kk!="per_anchor"}
with open("/tmp/wm_agentic_scale_results.json","w") as f:
    json.dump({"results":out,"winner":winner,
               "eeg":{"claims":gs.count_claims(),"communities":len(comms),"entities":len(ents)},
               "config":{"model":MN,"steps":STEPS,"bs":BS,"lr":LR,"ml":ML,"lora_r":32,
                          "domains":list(DOMAINS.keys()),"n_domain_dreams":len(all_dreams)},
               },f,indent=2,default=str)
print(f"Saved to /tmp/wm_agentic_scale_results.json")

os.makedirs("/tmp/wm_agentic_model",exist_ok=True)
m_pipe.save_pretrained("/tmp/wm_agentic_model")
tok.save_pretrained("/tmp/wm_agentic_model")
print(f"Best model saved to /tmp/wm_agentic_model")
store.close();gs.close()
pr("AGENTIC SCALE BENCHMARK COMPLETE")
