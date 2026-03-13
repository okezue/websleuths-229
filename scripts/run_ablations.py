"""Ablation study: measure contribution of each pipeline component.

Run on a Linux server:
    git clone <repo> && cd websleuths-229
    pip install -q -r requirements.txt
    python scripts/run_ablations.py --model meta-llama/Llama-3.2-1B

Ablation matrix (each row trains from the same base checkpoint):
    full          - dreaming + gating + content filter + dedup
    no_dream      - gating + filter + dedup  (dreaming OFF)
    no_gate       - dreaming + filter + dedup (gating OFF)
    no_filter     - dreaming + gating         (filter + dedup OFF)
    dream+gate    - dreaming + gating only    (confoundance check)
    dream+filter  - dreaming + filter + dedup (confoundance check)
    naive         - plain SFT, everything OFF
"""
from __future__ import annotations
import os,sys,copy,time,json,gc,random,logging,argparse,re
import torch
import numpy as np
from huggingface_hub import login

logging.basicConfig(level=logging.INFO,format="%(levelname)s %(name)s: %(message)s")
log=logging.getLogger("ablation")

# add project root to path
sys.path.insert(0,os.path.join(os.path.dirname(__file__),".."))

from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import get_peft_model,LoraConfig,TaskType
from datasets import Dataset

from wm.ingest.exa import ExaSrc
from wm.store import EpisodeStore
from wm.chunk import Chunker
from wm.dataset import DatasetBuilder
from wm.gate import EpisodeGate,exa_authority
from wm.recipe import EATRDRunner
from wm.eval import Evaluator,ProbeBuilder,AnchorEval
from wm.guard.drift import drift_kl
from wm.bench.eval_harness import DomainEvalHarness
from wm.search.content_filter import clean_text
from wm.search.dedup import dedup_chunks,dedup_raw
from wm.cfg import GateCfg,ChunkCfg,DatasetCfg

# config
MODEL_NAME=None
LORA_R=8
LORA_ALPHA=16
STEPS=int(os.environ.get("ABLATION_STEPS","100"))
BS=int(os.environ.get("ABLATION_BS","2"))
LR=2e-4
MAX_LEN=128
DREAM_N=2
DREAM_LEN=32
EVAL_N=int(os.environ.get("ABLATION_EVAL_N","20"))
DOMAIN_EVAL_N=int(os.environ.get("ABLATION_DOMAIN_N","20"))
EXA_KEY=os.environ.get("EXA_API_KEY","e337f35a-e56c-4ae7-8596-f44959053342")
HF_LLAMA_TOKEN="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"
_DEFAULT_RESULTS_ROOT=os.environ.get("WM_RESULTS_DIR",os.path.join(os.getcwd(),"results"))
_ABLATIONS_DIR=os.environ.get("ABLATIONS_DIR",os.path.join(_DEFAULT_RESULTS_ROOT,"ablations"))
OUT_PATH=None
CHECKPOINT_PATH=None
STATE_DIR=None

DREAM_PROMPTS=[
    "What is the capital of France?","Explain photosynthesis briefly.",
    "Who wrote Hamlet?","What causes earthquakes?",
    "How does the internet work?","What is DNA?",
    "Describe the water cycle.","What is machine learning?",
]
ANCHORS=[
    "The Earth revolves around the Sun.",
    "Water freezes at zero degrees Celsius.",
    "DNA carries genetic information.",
    "Gravity pulls objects toward Earth.",
    "The chemical formula for water is H2O.",
]
QUERIES=[
    "unsolved cold case forensic evidence 2024",
    "DNA forensic breakthroughs criminal investigation",
    "forensic science new techniques crime solving",
]

# ablation configs
# each is (name, dreaming_on, gating_on, filter_on)
ABLATIONS=[
    ("full",          True,  True,  True),
    ("no_dream",      False, True,  True),
    ("no_gate",       True,  False, True),
    ("no_filter",     True,  True,  False),
    ("dream+gate",    True,  True,  False),   # confoundance: same as no_filter
    ("dream+filter",  True,  False, True),    # confoundance: dreaming + filter, no gating
    ("naive",         False, False, False),
]
# note: dream+gate == no_filter by definition, but we keep both names
# so the results table reads naturally. We skip the duplicate run.

def _parse_args():
    ap=argparse.ArgumentParser(description="Run ablations for a specific model.")
    ap.add_argument("--model",required=True,
                    help="HF model id or local model path to evaluate")
    ap.add_argument("--out",default=os.environ.get("ABLATION_OUT"),
                    help="optional output JSON path; checkpoint path is derived from it")
    ap.add_argument("--domain-n",type=int,default=DOMAIN_EVAL_N,
                    help="samples per domain/MMLU benchmark (default: %(default)s)")
    return ap.parse_args()

def _sanitize_model_name(model_name):
    safe=re.sub(r"[^A-Za-z0-9._-]+","_",model_name.strip())
    return safe.strip("._-") or "model"

def _login_for_llama_model(model_name):
    if "llama" in model_name.lower():
        login(HF_LLAMA_TOKEN)

def _default_out_path(model_name):
    fname=f"ablation_results_{_sanitize_model_name(model_name)}.json"
    return os.path.join(_ABLATIONS_DIR,fname)

def _configure_run(args):
    global MODEL_NAME,OUT_PATH,CHECKPOINT_PATH,STATE_DIR,DOMAIN_EVAL_N
    MODEL_NAME=args.model
    _login_for_llama_model(MODEL_NAME)
    DOMAIN_EVAL_N=args.domain_n
    OUT_PATH=args.out or _default_out_path(MODEL_NAME)
    out_dir=os.path.dirname(OUT_PATH) or "."
    out_stem,out_ext=os.path.splitext(os.path.basename(OUT_PATH))
    if not out_ext:
        out_ext=".json"
        OUT_PATH=os.path.join(out_dir,f"{out_stem}{out_ext}")
    CHECKPOINT_PATH=os.path.join(out_dir,f"{out_stem}_checkpoint{out_ext}")
    STATE_DIR=os.path.join(out_dir,f"{out_stem}_state")

def _save_checkpoint(results):
    """Save completed results so we can resume after disconnect."""
    os.makedirs(os.path.dirname(CHECKPOINT_PATH) or ".",exist_ok=True)
    out={}
    for k,v in results.items():
        out[k]={kk:vv for kk,vv in v.items() if kk!="per_anchor"}
    with open(CHECKPOINT_PATH,"w") as f:
        json.dump(out,f,indent=2,default=str)
    log.info("checkpoint saved: %d configs -> %s",len(out),CHECKPOINT_PATH)

def _adapter_ckpt_path(name):
    os.makedirs(STATE_DIR,exist_ok=True)
    safe=_sanitize_model_name(name)
    return os.path.join(STATE_DIR,f"{safe}_adapter.pt")

def _save_adapter_checkpoint(model,name):
    path=_adapter_ckpt_path(name)
    state={k:v.detach().cpu() for k,v in model.state_dict().items() if "lora_" in k}
    torch.save(state,path)
    return path

def _load_adapter_checkpoint(model,path):
    state=torch.load(path,map_location="cpu")
    incompat=model.load_state_dict(state,strict=False)
    if incompat.unexpected_keys:
        log.warning("adapter checkpoint had unexpected keys: %s",incompat.unexpected_keys[:5])
    non_lora_missing=[k for k in incompat.missing_keys if "lora_" in k]
    if non_lora_missing:
        log.warning("adapter checkpoint missing LoRA keys: %s",non_lora_missing[:5])

def _checkpoint_result(results,name,metrics,status):
    payload=dict(metrics)
    payload["_status"]=status
    results[name]=payload
    _save_checkpoint(results)

def _load_checkpoint():
    """Load previous checkpoint if it exists."""
    if not os.path.exists(CHECKPOINT_PATH):
        return {}
    try:
        with open(CHECKPOINT_PATH) as f:
            data=json.load(f)
        log.info("resumed checkpoint: %d configs from %s",len(data),CHECKPOINT_PATH)
        return data
    except Exception:
        return {}

def pr(msg):
    print(f"\n{'='*60}\n{msg}\n{'='*60}")

def gpu_info():
    if not torch.cuda.is_available():
        return "CPU only"
    name=torch.cuda.get_device_name(0)
    mem=torch.cuda.get_device_properties(0).total_memory/1e9
    return f"{name} ({mem:.1f} GB)"

def gpu_mem():
    if not torch.cuda.is_available():
        return ""
    used=torch.cuda.memory_allocated()/1e9
    peak=torch.cuda.max_memory_allocated()/1e9
    return f" [GPU: {used:.1f}/{peak:.1f} GB]"

def fmt_eta(seconds):
    if seconds<60:return f"{seconds:.0f}s"
    m,s=divmod(int(seconds),60)
    if m<60:return f"{m}m{s:02d}s"
    h,m=divmod(m,60)
    return f"{h}h{m:02d}m"

def make_model(mn=None):
    mn=mn or MODEL_NAME
    tok=AutoTokenizer.from_pretrained(mn,trust_remote_code=True)
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model=AutoModelForCausalLM.from_pretrained(mn,torch_dtype=dtype,
                                                trust_remote_code=True)
    if torch.cuda.is_available():
        model=model.cuda()
    lc=LoraConfig(r=LORA_R,lora_alpha=LORA_ALPHA,lora_dropout=0.05,
        target_modules=["q_proj","v_proj"],task_type=TaskType.CAUSAL_LM)
    model=get_peft_model(model,lc)
    return model,tok

def fetch_data():
    """Fetch training data from Exa. Returns list of episodes."""
    if not EXA_KEY:
        log.warning("No EXA_API_KEY set. Using synthetic fallback data.")
        return _synthetic_data()
    os.environ["EXA_API_KEY"]=EXA_KEY
    src=ExaSrc()
    all_eps=[]
    for q in QUERIES:
        try:
            eps=src.fetch(q,n=5)
            eps=exa_authority(eps)
            all_eps.extend(eps)
            log.info("'%s': %d episodes",q,len(eps))
        except Exception as e:
            log.warning("'%s' failed: %s",q,e)
    return all_eps

def _synthetic_data():
    """Fallback: build minimal synthetic episodes for testing."""
    from wm.types import Episode
    texts=[
        "Forensic DNA analysis has revolutionized criminal investigations. Modern techniques like touch DNA and familial searching allow investigators to identify suspects from trace amounts of biological material left at crime scenes.",
        "Cold case investigations have seen renewed interest with advances in genetic genealogy. Investigators can now use public DNA databases to identify suspects in decades-old unsolved cases through distant relative matching.",
        "The development of rapid DNA technology enables crime labs to process DNA samples in under two hours, compared to the traditional timeline of weeks or months. This has significant implications for both active investigations and backlog processing.",
        "Forensic science continues to evolve with new methods for analyzing digital evidence, including smartphone forensics, cloud data extraction, and AI-assisted pattern recognition in surveillance footage.",
        "Recent advances in forensic toxicology include more sensitive detection methods for novel psychoactive substances and improved techniques for post-mortem drug analysis in decomposed remains.",
    ]
    eps=[]
    for i,t in enumerate(texts):
        eps.append(Episode(url=f"https://synthetic.example.com/article-{i}",
                           title=f"Forensic Science Article {i}",body=t,
                           authority=0.8))
    return eps

def prepare_data(episodes,use_filter=True,use_dedup=True):
    """Chunk episodes into train/eval datasets. Optionally apply filter + dedup."""
    gate=EpisodeGate(GateCfg(min_sources=1,min_consistency=0.0,uncertainty_thresh=0.0))
    gate.check(episodes)

    if use_filter:
        filtered=[]
        for ep in episodes:
            cleaned=clean_text(ep.body)
            if cleaned:
                ep_copy=copy.copy(ep)
                ep_copy.body=cleaned
                filtered.append(ep_copy)
            else:
                filtered.append(ep)
        episodes=filtered

    store=EpisodeStore("/tmp/wm_ablation_ep.db")
    store.put_many(episodes)
    chunks=Chunker(ChunkCfg(max_tok=256,overlap=32)).chunk_many(store.all())

    if use_dedup:
        raw_dicts=[{"url":ep.url,"text":ep.body} for ep in episodes]
        raw_dicts=dedup_raw(raw_dicts)
        chunks=dedup_chunks(chunks)

    log.info("chunks: %d (filter=%s, dedup=%s)",len(chunks),use_filter,use_dedup)

    ds=DatasetBuilder(DatasetCfg(recipe="cpt")).build(chunks)
    probes=ProbeBuilder().build_all(chunks)
    store.close()
    os.remove("/tmp/wm_ablation_ep.db") if os.path.exists("/tmp/wm_ablation_ep.db") else None
    return ds,probes,chunks

def eval_model(model,tok,ds_eval,probes,tag=""):
    """Eval: perplexity, accuracy, probe scores, anchor NLL."""
    ev=Evaluator(model,tok,max_len=128)
    report=ev.evaluate(ds_eval)
    pscore=ev.evaluate_probes(probes[:10])
    anc=AnchorEval(model,tok,ANCHORS)
    anll=anc.nll()
    per_anc=anc.per_anchor_nll()
    print(f"  [{tag}] ppl={report.ppl:.2f} acc={report.acc:.4f} "
          f"probe_f1={pscore['mean_f1']:.4f} anchor_nll={anll:.4f}")
    return {"ppl":report.ppl,"acc":report.acc,"hit_rate":pscore["hit_rate"],
            "mean_f1":pscore["mean_f1"],"anchor_nll":anll,
            "per_anchor":per_anc,"n_probes":pscore["n_probes"]}

def eval_domains(model,tok,n=DOMAIN_EVAL_N,existing=None,on_update=None):
    """Run domain benchmark evals (HF datasets, no API keys needed)."""
    harness=DomainEvalHarness(model,tok,n_samples=n)
    scores=dict(existing or {})
    for domain in ["finance","legal","chemistry","medicine"]:
        if domain in scores and scores[domain].get("n",0)>0:
            print(f"    {domain}: resumed acc={scores[domain]['acc']:.4f} (n={scores[domain]['n']})")
            continue
        try:
            bs=harness.eval_domain(domain,n=n)
            scores[domain]={"acc":bs.acc,"n":bs.n}
            print(f"    {domain}: acc={bs.acc:.4f} (n={bs.n})")
        except Exception as e:
            log.warning("domain %s failed: %s",domain,e)
            scores[domain]={"acc":0.0,"n":0}
        if on_update:
            on_update(scores)
    for domain in ["finance","legal","chemistry","medicine"]:
        key=f"mmlu_{domain}"
        if key in scores and scores[key].get("n",0)>0:
            print(f"    {key}: resumed acc={scores[key]['acc']:.4f} (n={scores[key]['n']})")
            continue
        try:
            ms=harness.eval_mmlu(domain,n=n)
            scores[key]={"acc":ms.acc,"n":ms.n}
            print(f"    {key}: acc={ms.acc:.4f} (n={ms.n})")
        except Exception as e:
            log.warning("mmlu %s failed: %s",domain,e)
            scores[key]={"acc":0.0,"n":0}
        if on_update:
            on_update(scores)
    return scores

def run_ablation(name,base_model,base_snap,tok,ds,ds_eval,probes,
                 dream_on,gate_on,filter_on,resume_state=None,results=None):
    """Run one ablation config: train + eval."""
    pr(f"ABLATION: {name} (dream={dream_on}, gate={gate_on}, filter={filter_on})")
    t0=time.time()
    resume_state=resume_state or {}
    results=results if results is not None else {}
    status=resume_state.get("_status","")

    model=copy.deepcopy(base_model)
    model.load_state_dict(base_snap)
    metrics={k:v for k,v in resume_state.items() if k!="per_anchor"}
    metrics["config"]={"dream":dream_on,"gate":gate_on,"filter":filter_on}

    if status in {"trained","core_eval","drift","domains_partial"}:
        ckpt=resume_state.get("_adapter_ckpt")
        if ckpt and os.path.exists(ckpt):
            _load_adapter_checkpoint(model,ckpt)
            print(f"  resumed trained adapter from {ckpt}")
        else:
            print("  partial checkpoint missing adapter weights; retraining config")
            status=""

    if not status:
        teacher=copy.deepcopy(model).eval()

        # dreaming: lam_init=1.0 if on, 0.0 if off
        lam=1.0 if dream_on else 0.0
        runner=EATRDRunner(
            lr=LR,max_steps=STEPS,bs=BS,temp=2.0,
            eps_min=0.01,alpha=0.5,rho=0.01,lam_init=lam,
            max_len=MAX_LEN,dream_n=DREAM_N,dream_len=DREAM_LEN,
            d_targ=0.005)

        tr=runner.run(model,teacher,ds,DREAM_PROMPTS,tok)
        print(f"  train: loss={tr.loss:.4f} dream_loss={tr.dream_loss:.4f} "
              f"steps={tr.steps} lam={tr.extras.get('lambda',0):.4f}")
        del teacher;gc.collect()
        metrics["train_loss"]=tr.loss
        metrics["dream_loss"]=tr.dream_loss
        metrics["_adapter_ckpt"]=_save_adapter_checkpoint(model,name)
        _checkpoint_result(results,name,metrics,"trained")
        status="trained"

    # eval
    if status=="trained":
        eval_metrics=eval_model(model,tok,ds_eval,probes,tag=name)
        metrics.update(eval_metrics)
        _checkpoint_result(results,name,metrics,"core_eval")
        status="core_eval"

    # drift
    if status=="core_eval":
        dk=drift_kl(model,base_model,tok,ANCHORS[:3],max_len=64)
        metrics["drift_kl"]=dk
        print(f"  drift_kl={dk:.6f}")
        _checkpoint_result(results,name,metrics,"drift")
        status="drift"

    # domain benchmarks
    if status in {"drift","domains_partial"}:
        partial_domains=metrics.get("domains",{})
        def _on_domain_update(scores):
            metrics["domains"]=copy.deepcopy(scores)
            _checkpoint_result(results,name,metrics,"domains_partial")
        domain_scores=eval_domains(model,tok,n=DOMAIN_EVAL_N,
                                   existing=partial_domains,on_update=_on_domain_update)
        metrics["domains"]=domain_scores

    metrics["time"]=time.time()-t0
    metrics.pop("_status",None)

    del model;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    _checkpoint_result(results,name,metrics,"done")
    return results[name]

def sub_ds(ds,n=EVAL_N):
    if len(ds)<=n:return ds
    idx=list(range(len(ds)))
    random.seed(42)
    random.shuffle(idx)
    return ds.select(idx[:n])

def print_results_table(results):
    """Print formatted results table."""
    pr("RESULTS TABLE")
    hdr=(f"{'Config':<16}|{'PPL':>7}|{'Acc':>7}|{'F1':>6}|{'AncNLL':>7}|"
         f"{'Drift':>8}|{'Loss':>7}|{'Dream':>6}|{'Time':>6}")
    print(hdr)
    print("-"*len(hdr))
    for name,m in results.items():
        print(f"{name:<16}|{m.get('ppl',0):>7.2f}|{m.get('acc',0):>7.4f}|"
              f"{m.get('mean_f1',0):>6.3f}|{m.get('anchor_nll',0):>7.3f}|"
              f"{m.get('drift_kl',0):>8.5f}|{m.get('train_loss',0):>7.4f}|"
              f"{m.get('dream_loss',0):>6.3f}|{m.get('time',0):>5.0f}s")

def print_domain_table(results):
    """Print domain benchmark comparison."""
    pr("DOMAIN BENCHMARKS")
    domains=["finance","legal","chemistry","medicine",
             "mmlu_finance","mmlu_legal","mmlu_chemistry","mmlu_medicine"]
    hdr=f"{'Config':<16}|"+"|".join(f"{d:>12}" for d in domains)
    print(hdr)
    print("-"*len(hdr))
    for name,m in results.items():
        ds=m.get("domains",{})
        vals=[]
        for d in domains:
            acc=ds.get(d,{}).get("acc",0.0)
            vals.append(f"{acc:>12.4f}")
        print(f"{name:<16}|"+"|".join(vals))

def print_ablation_analysis(results):
    """Print component contribution analysis."""
    pr("COMPONENT CONTRIBUTION ANALYSIS")
    if "full" not in results or "naive" not in results:
        print("Need both 'full' and 'naive' runs for analysis.")
        return

    full=results["full"]
    naive=results["naive"]
    print(f"Full system F1:  {full['mean_f1']:.4f}")
    print(f"Naive SFT F1:    {naive['mean_f1']:.4f}")
    print(f"Total gain:      {full['mean_f1']-naive['mean_f1']:+.4f}")
    print()

    # single-component contribution = full - (full without component)
    components={"dreaming":"no_dream","gating":"no_gate","filter+dedup":"no_filter"}
    print("Single-component contributions (full - ablated):")
    for comp,ablated_name in components.items():
        if ablated_name in results:
            delta=full["mean_f1"]-results[ablated_name]["mean_f1"]
            drift_delta=results[ablated_name]["drift_kl"]-full["drift_kl"]
            print(f"  {comp:<16}: dF1={delta:+.4f}  dDrift={drift_delta:+.6f}")

    # confoundance: does combining two components explain more than sum of parts?
    print("\nConfoundance check:")
    if "dream+gate" in results and "no_dream" in results and "no_gate" in results:
        # contribution of dream alone = full - no_dream
        dream_alone=full["mean_f1"]-results["no_dream"]["mean_f1"]
        # contribution of gate alone = full - no_gate
        gate_alone=full["mean_f1"]-results["no_gate"]["mean_f1"]
        # dream+gate combined (no filter) vs naive
        combined=results["dream+gate"]["mean_f1"]-naive["mean_f1"]
        sum_parts=dream_alone+gate_alone
        interaction=combined-sum_parts
        print(f"  dream contribution:    {dream_alone:+.4f}")
        print(f"  gate contribution:     {gate_alone:+.4f}")
        print(f"  dream+gate combined:   {combined:+.4f} (vs naive)")
        print(f"  sum of parts:          {sum_parts:+.4f}")
        print(f"  interaction effect:    {interaction:+.4f}")
        if abs(interaction)>0.01:
            print(f"  -> NON-TRIVIAL interaction between dreaming and gating")
        else:
            print(f"  -> Components appear INDEPENDENT (small interaction)")

    if "dream+filter" in results:
        dream_alone=full["mean_f1"]-results["no_dream"]["mean_f1"]
        filter_alone=full["mean_f1"]-results["no_filter"]["mean_f1"]
        combined=results["dream+filter"]["mean_f1"]-naive["mean_f1"]
        sum_parts=dream_alone+filter_alone
        interaction=combined-sum_parts
        print(f"\n  dream+filter combined: {combined:+.4f} (vs naive)")
        print(f"  sum of parts:          {sum_parts:+.4f}")
        print(f"  interaction effect:    {interaction:+.4f}")

def main():
    args=_parse_args()
    _configure_run(args)
    prev=_load_checkpoint()
    pr("ABLATION STUDY")
    print(f"Model: {MODEL_NAME}")
    print(f"GPU: {gpu_info()}")
    print(f"Steps: {STEPS}, BS: {BS}, LR: {LR}")
    print(f"Domain eval samples: {DOMAIN_EVAL_N}")
    n_unique=len(set((d,g,f) for _,d,g,f in ABLATIONS))
    print(f"Ablation matrix: {len(ABLATIONS)} configs ({n_unique} unique runs + baseline)")

    # fetch data once
    pr("PHASE 1: FETCH DATA")
    episodes=fetch_data()
    print(f"Total episodes: {len(episodes)}")

    # prepare two versions of the data: filtered and unfiltered
    pr("PHASE 2: PREPARE DATA")
    ds_filtered,probes_filtered,chunks_filtered=prepare_data(
        episodes,use_filter=True,use_dedup=True)
    ds_raw,probes_raw,chunks_raw=prepare_data(
        episodes,use_filter=False,use_dedup=False)

    ds_eval_filtered=sub_ds(ds_filtered)
    ds_eval_raw=sub_ds(ds_raw)
    print(f"Filtered: {len(ds_filtered)} rows, {len(probes_filtered)} probes")
    print(f"Raw:      {len(ds_raw)} rows, {len(probes_raw)} probes")

    # load model + baseline
    pr("PHASE 3: LOAD MODEL + BASELINE")
    t0=time.time()
    base_model,tok=make_model()
    base_model.print_trainable_parameters()
    base_snap={k:v.clone() for k,v in base_model.state_dict().items()}

    if "baseline" in prev and (
        prev["baseline"].get("_status")=="done" or
        ("mean_f1" in prev["baseline"] and "domains" in prev["baseline"])
    ):
        baseline=prev["baseline"]
        print("Baseline: RESUMED from checkpoint")
    else:
        # baseline eval (no training)
        print("Computing baseline (no training)...")
        baseline=eval_model(base_model,tok,ds_eval_filtered,probes_filtered,"baseline")
        baseline["drift_kl"]=0.0
        baseline["train_loss"]=0.0
        baseline["dream_loss"]=0.0
        baseline["time"]=time.time()-t0
        baseline["domains"]=eval_domains(base_model,tok,n=DOMAIN_EVAL_N)
        baseline["config"]={"dream":False,"gate":False,"filter":False}

    results={"baseline":baseline}
    _save_checkpoint(results)

    # run ablations
    pr("PHASE 4: RUN ABLATIONS")
    seen_configs=set()
    run_times=[]
    runs_done=0
    total_runs=len(set((d,g,f) for _,d,g,f in ABLATIONS))
    study_t0=time.time()

    for name,dream,gate,filt in ABLATIONS:
        config_key=(dream,gate,filt)

        # dream+gate has same config as no_filter, alias the result
        if config_key in seen_configs:
            for prev_name,prev_m in results.items():
                pc=prev_m.get("config",{})
                if (pc.get("dream"),pc.get("gate"),pc.get("filter"))==config_key:
                    results[name]=prev_m
                    print(f"\n  {name}: same config as {prev_name}, reusing result")
                    break
            continue
        seen_configs.add(config_key)

        # skip if already completed in a previous run
        if name in prev and (
            prev[name].get("_status")=="done" or
            ("mean_f1" in prev[name] and "domains" in prev[name])
        ):
            results[name]=prev[name]
            runs_done+=1
            print(f"\n  [{runs_done}/{total_runs}] {name}: RESUMED from checkpoint")
            continue

        # ETA
        runs_done+=1
        if run_times:
            avg_t=sum(run_times)/len(run_times)
            remaining=(total_runs-runs_done)*avg_t
            print(f"\n  [{runs_done}/{total_runs}] ETA: ~{fmt_eta(remaining)}{gpu_mem()}")
        else:
            print(f"\n  [{runs_done}/{total_runs}] First run...{gpu_mem()}")

        # pick the right dataset based on filter setting
        if filt:
            ds,ds_eval,probes=ds_filtered,ds_eval_filtered,probes_filtered
        else:
            ds,ds_eval,probes=ds_raw,ds_eval_raw,probes_raw

        m=run_ablation(name,base_model,base_snap,tok,ds,ds_eval,probes,
                       dream,gate,filt,resume_state=prev.get(name),results=results)
        results[name]=m
        run_times.append(m["time"])

    total_time=time.time()-study_t0
    print(f"\nAll ablations done in {fmt_eta(total_time)}{gpu_mem()}")

    # results
    print_results_table(results)
    print_domain_table(results)
    print_ablation_analysis(results)

    # save
    out={}
    for k,v in results.items():
        out[k]={kk:vv for kk,vv in v.items() if kk!="per_anchor" and not kk.startswith("_")}
    os.makedirs(os.path.dirname(OUT_PATH) or ".",exist_ok=True)
    with open(OUT_PATH,"w") as f:
        json.dump({"results":out,"model":MODEL_NAME,"steps":STEPS,
                   "ablations":[a[0] for a in ABLATIONS]},f,indent=2,default=str)
    print(f"\nSaved to {OUT_PATH}")
    pr("ABLATION STUDY COMPLETE")

if __name__=="__main__":
    main()
