#!/usr/bin/env python3
import os,sys,time,json,argparse,logging
logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(name)s %(message)s",
                    datefmt="%H:%M:%S",stream=sys.stdout)
os.environ["TOKENIZERS_PARALLELISM"]="false"
import torch
from transformers import AutoTokenizer
from datasets import Dataset
from wm.cfg import WMCfg,SearchCfg,GraphCfg,GuardCfg,SearchGateCfg,UpdateGateCfg,PaceCfg,IterCLCfg
from wm.recipe.tinker_eatrd import TinkerEATRDRunner
from wm.search.agent import AgenticSearcher
from wm.graph.store import GraphStore
from wm.graph.dream_gen import gen_dream_prompts,gen_dream_bank
from wm.graph.community import detect_communities
from wm.bench.iterative import TOPIC_SCHEDULE
from wm.adapt import adaptive_steps as compute_adaptive_steps

log=logging.getLogger("tinker_bench")
DOMAINS=["finance","legal","chemistry","medicine"]

def pr(msg):
    s=f"\n{'='*70}\n{msg}\n{'='*70}"
    print(s,flush=True);log.info(msg)

def tinker_eval_mmlu(runner,tok,domain:str,n:int=50)->dict:
    from datasets import load_dataset
    _map={"finance":"professional_accounting","legal":"professional_law",
          "chemistry":"college_chemistry","medicine":"professional_medicine"}
    subj=_map.get(domain)
    if not subj:return {"acc":0,"n":0}
    ds=None
    for sp in ["test","validation"]:
        try:ds=load_dataset("cais/mmlu",subj,split=sp);break
        except:pass
    if ds is None:return {"acc":0,"n":0}
    import random
    rng=random.Random(42)
    idx=list(range(len(ds)))
    if n>0 and n<len(ds):
        rng.shuffle(idx)
        idx=idx[:n]
    letters=["A","B","C","D"]
    cor,tot=0,0
    for i in idx:
        row=ds[i]
        q=row.get("question","")
        choices=row.get("choices",[])
        ans=row.get("answer",0)
        if not choices:tot+=1;continue
        prompt=f"Q: {q}\n"
        for ci,c in enumerate(choices[:4]):
            prompt+=f"{letters[ci]}) {c}\n"
        prompt+="Answer:"
        try:
            resp=runner.sample(prompt,tok,max_tokens=64,temp=0.01)
            import re as _re
            clean=_re.sub(r'<think>.*?</think>','',resp,flags=_re.DOTALL).strip()
            if not clean:clean=resp.strip()
            pred=""
            for ch in clean:
                if ch.upper() in letters:
                    pred=ch.upper();break
        except:
            pred=""
        if isinstance(ans,int) and ans<4:gold=letters[ans]
        elif isinstance(ans,str) and ans.upper() in letters:gold=ans.upper()
        else:gold=""
        if pred==gold:cor+=1
        tot+=1
    return {"acc":cor/max(tot,1),"n":tot}

def tinker_generate_examples(runner,tok,domain:str,n:int=3)->list[dict]:
    prompts={
        "finance":["Explain the price-to-earnings ratio and its significance in stock valuation.",
                   "What are the key components of a company's balance sheet?",
                   "Describe how credit risk is assessed under Basel IV."],
        "legal":["What is the significance of the Fourth Amendment in digital privacy?",
                 "Explain the doctrine of stare decisis in constitutional law.",
                 "How does intellectual property law apply to AI-generated content?"],
        "chemistry":["Describe the SN2 reaction mechanism in organic chemistry.",
                    "What is the role of CRISPR-Cas9 in gene editing?",
                    "Explain molecular dynamics simulation in computational chemistry."],
        "medicine":["What are common drug-drug interactions in clinical pharmacology?",
                   "Describe the pathophysiology of heart failure.",
                   "How do mRNA vaccines elicit an immune response?"],
    }
    dp=prompts.get(domain,prompts["finance"])[:n]
    out=[]
    for p in dp:
        try:
            r=runner.sample(p,tok,max_tokens=200,temp=0.7)
        except:
            r="[generation failed]"
        out.append({"prompt":p,"response":r})
    return out

def main():
    ap=argparse.ArgumentParser(description="Tinker-backed EATRD benchmark (bigger models)")
    ap.add_argument("--model",default="Qwen/Qwen2.5-7B")
    ap.add_argument("--lora-r",type=int,default=32)
    ap.add_argument("--out",default="/tmp/wm_tinker_bench")
    ap.add_argument("--exa-key",default=os.environ.get("EXA_API_KEY",""))
    ap.add_argument("--anthropic-key",default=os.environ.get("ANTHROPIC_API_KEY",""))
    ap.add_argument("--tinker-key",default=os.environ.get("TINKER_API_KEY",""))
    ap.add_argument("--hf-token",default=os.environ.get("HF_TOKEN",""))
    ap.add_argument("--lr",type=float,default=2e-4)
    ap.add_argument("--bs",type=int,default=4)
    ap.add_argument("--ml",type=int,default=256)
    ap.add_argument("--steps",type=int,default=100)
    ap.add_argument("--eval-n",type=int,default=50,help="MMLU sample size per domain (0=full)")
    ap.add_argument("--mmlu-every",type=int,default=2)
    args=ap.parse_args()
    if args.hf_token:os.environ["HF_TOKEN"]=args.hf_token
    if args.tinker_key:os.environ["TINKER_API_KEY"]=args.tinker_key

    od=args.out
    os.makedirs(f"{od}/plots",exist_ok=True)
    fh=logging.FileHandler(f"{od}/bench.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s",
                                       datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(fh)

    pr(f"TINKER BENCHMARK: {args.model} lora_r={args.lora_r}")
    log.info("backend: Tinker (remote GPUs)")
    tok=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True)
    if tok.pad_token is None:tok.pad_token=tok.eos_token

    runner=TinkerEATRDRunner(
        model_name=args.model,lora_rank=args.lora_r,
        lr=args.lr,max_steps=args.steps,bs=args.bs,max_len=args.ml,
        lam_init=1.0,d_targ=0.5,lam_floor=0.01,lam_ceil=10.0,
        pi_warmup_frac=0.15,dream_n=4,
        tinker_api_key=args.tinker_key)

    extraction_backend="claude" if args.anthropic_key else "regex"
    cfg=WMCfg(
        search=SearchCfg(exa_api_key=args.exa_key,max_rounds=5,queries_per_round=6,
                         min_claims=5,res_per_query=5,
                         extraction_backend=extraction_backend,
                         anthropic_api_key=args.anthropic_key),
        graph=GraphCfg(db_path=f"{od}/graph.db"),
        guard=GuardCfg(enabled=False),
        search_gate=SearchGateCfg(enabled=False),
        update_gate=UpdateGateCfg(enabled=False),
        pace=PaceCfg(web_budget=500,ft_budget=100),
        iter_cl=IterCLCfg(min_steps=args.steps,max_steps=args.steps))

    report={"config":{"model":args.model,"lora_r":args.lora_r,
                       "lr":args.lr,"bs":args.bs,"steps":args.steps,
                       "backend":"tinker","extraction":extraction_backend}}

    pr("INITIALIZING TINKER CLIENTS")
    runner.init_clients()

    pr("PHASE 1: BASELINE EVAL (pre-training)")
    bl_mmlu={}
    bl_exs={}
    for d in DOMAINS:
        log.info("  baseline MMLU_%s (n=%d)...",d,args.eval_n)
        bl_mmlu[d]=tinker_eval_mmlu(runner,tok,d,n=args.eval_n)
        log.info("  MMLU_%s: acc=%.4f n=%d",d,bl_mmlu[d]["acc"],bl_mmlu[d]["n"])
        bl_exs[d]=tinker_generate_examples(runner,tok,d,n=3)
        for ex in bl_exs[d]:
            log.info("    Q: %s",ex["prompt"][:80])
            log.info("    A: %s",ex["response"][:150])
    report["baseline"]={"mmlu":bl_mmlu,"examples":bl_exs}

    pr("PHASE 2: SEQUENTIAL DOMAIN LEARNING")
    mmlu_prog={d:[bl_mmlu[d]["acc"]] for d in DOMAINS}
    prev_doms=[]
    report["domains"]={}
    total_ft_time=0
    gs=GraphStore(cfg.graph.db_path)

    for di,dom in enumerate(DOMAINS):
        pr(f"DOMAIN {di+1}/{len(DOMAINS)}: {dom.upper()}")
        topics=TOPIC_SCHEDULE.get(dom,[])
        dom_report={"topics":[]}
        dbank=None

        for ti,topic in enumerate(topics):
            log.info("[%d/%d] topic: %s",ti+1,len(topics),topic)

            log.info("  searching...")
            t0=time.time()
            searcher=AgenticSearcher(cfg.search)
            sr=searcher.search(topic)
            gs.store_result(sr)
            search_time=time.time()-t0
            log.info("  search: %d claims, %d communities, %d chunks in %.1fs",
                     len(sr.claims),len(sr.communities),len(sr.chunks),search_time)

            if dbank is None:
                dbank=gen_dream_bank(sr.communities,sr.claims,tok)
            else:
                dbank.add_episode(sr.communities,sr.claims)
            dreams=gen_dream_prompts(sr.communities,sr.claims)

            if sr.train_rows:
                ds_rows=sr.train_rows
                log.info("  using %d Claude KG train_rows",len(ds_rows))
            else:
                ds_rows=[{"text":c.text,"authority":c.confidence} for c in sr.claims[:100]]
                if not ds_rows:
                    ds_rows=[{"text":c.text,"authority":c.authority} for c in sr.chunks]
                log.info("  using %d fallback train rows",len(ds_rows))

            if not ds_rows:
                log.warning("  no training data, skipping topic")
                dom_report["topics"].append({"topic":topic,"action":"no_data"})
                continue

            ds=Dataset.from_list(ds_rows)
            log.info("  training on Tinker (%s)...",args.model)
            t1=time.time()
            tr=runner.run(ds,dreams,tok,dbank=dbank)
            ft_time=time.time()-t1
            total_ft_time+=ft_time
            log.info("  TRAINED: loss=%.4f dream=%.4f steps=%d time=%.1fs",
                     tr.loss,tr.dream_loss or 0,tr.steps,ft_time)
            if tr.history:
                log.info("  loss: %.4f -> %.4f  lambda: %.4f -> %.4f",
                         tr.history[0]["loss"],tr.history[-1]["loss"],
                         tr.history[0]["lambda"],tr.history[-1]["lambda"])

            tr_dict={"topic":topic,"domain":dom,"search_time":search_time,
                     "ft_time":ft_time,"loss":tr.loss,"dream_loss":tr.dream_loss,
                     "steps":tr.steps,"claims":len(sr.claims),
                     "communities":len(sr.communities),
                     "train_rows":len(ds_rows)}
            if tr.history:
                tr_dict["loss_start"]=tr.history[0]["loss"]
                tr_dict["loss_end"]=tr.history[-1]["loss"]
                tr_dict["lambda_start"]=tr.history[0]["lambda"]
                tr_dict["lambda_end"]=tr.history[-1]["lambda"]

            if (ti+1)%args.mmlu_every==0 or ti==len(topics)-1:
                log.info("  MMLU checkpoint...")
                ms=tinker_eval_mmlu(runner,tok,dom,n=args.eval_n)
                mmlu_prog[dom].append(ms["acc"])
                delta=ms["acc"]-bl_mmlu[dom]["acc"]
                log.info("  MMLU_%s: acc=%.4f (delta=%+.4f vs baseline)",dom,ms["acc"],delta)
                tr_dict["mmlu_checkpoint"]=ms

            dom_report["topics"].append(tr_dict)

        prev_doms.append(dom)
        pr(f"DOMAIN {dom.upper()} COMPLETE — POST-DOMAIN EVAL")

        dm={}
        for d2 in DOMAINS:
            ms=tinker_eval_mmlu(runner,tok,d2,n=args.eval_n)
            dm[d2]=ms
            mmlu_prog[d2].append(ms["acc"])
            delta=ms["acc"]-bl_mmlu[d2]["acc"]
            log.info("  MMLU_%s: acc=%.4f (delta=%+.4f)",d2,ms["acc"],delta)
        dom_report["mmlu_full"]=dm

        dom_exs=tinker_generate_examples(runner,tok,dom,n=3)
        dom_report["examples_after"]=dom_exs
        log.info("  examples after %s:",dom)
        for i,ex in enumerate(dom_exs):
            log.info("    Q: %s",ex["prompt"][:80])
            log.info("    A: %s",ex["response"][:150])
            if i<len(bl_exs.get(dom,[])):
                log.info("    BEFORE: %s",bl_exs[dom][i]["response"][:150])

        report["domains"][dom]=dom_report
        with open(f"{od}/results_partial.json","w") as f:
            json.dump(report,f,indent=2,default=str)
        log.info("  partial results saved")

    pr("PHASE 3: FINAL EVAL")
    fn_mmlu={}
    fn_exs={}
    for d in DOMAINS:
        fn_mmlu[d]=tinker_eval_mmlu(runner,tok,d,n=args.eval_n)
        fn_exs[d]=tinker_generate_examples(runner,tok,d,n=3)
        delta=fn_mmlu[d]["acc"]-bl_mmlu[d]["acc"]
        log.info("  final MMLU_%s: acc=%.4f (delta=%+.4f)",d,fn_mmlu[d]["acc"],delta)
    report["final"]={"mmlu":fn_mmlu,"examples":fn_exs}
    report["summary"]={"total_ft_time":total_ft_time,"mmlu_prog":mmlu_prog}

    with open(f"{od}/results.json","w") as f:
        json.dump(report,f,indent=2,default=str)

    pr("RESULTS")
    print(f"Model: {args.model} (Tinker)  LoRA r={args.lora_r}")
    print(f"Total FT time: {total_ft_time:.1f}s")
    print(f"\n{'MMLU':<22} {'Baseline':>10} {'Final':>10} {'Delta':>10}")
    print("-"*54)
    for d in DOMAINS:
        b=bl_mmlu[d]["acc"];f_=fn_mmlu[d]["acc"]
        print(f"  {d:<20} {b:>10.4f} {f_:>10.4f} {f_-b:>+10.4f}")
    print(f"\nResults: {od}/results.json")
    gs.close()
    pr("TINKER BENCHMARK COMPLETE")

if __name__=="__main__":
    main()
