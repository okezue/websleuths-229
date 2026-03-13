#!/usr/bin/env python3
import os,sys,time,json,gc,argparse,logging
logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(name)s %(message)s",
                    datefmt="%H:%M:%S",stream=sys.stdout)
os.environ["TOKENIZERS_PARALLELISM"]="false"
import torch

log=logging.getLogger("circuit_bench")

DOMAIN_PROBES={
    "finance":[
        "The price-to-earnings ratio is calculated by",
        "A company's balance sheet shows total assets of",
        "The SEC regulates financial markets by",
        "Credit risk under Basel IV is measured using",
        "Options pricing uses the Black-Scholes model to",
    ],
    "legal":[
        "The Fourth Amendment protects citizens from",
        "Stare decisis means that courts must",
        "Intellectual property law covers patents,",
        "The Supreme Court ruled in Miranda v Arizona that",
        "Due process under the Fifth Amendment requires",
    ],
    "chemistry":[
        "The SN2 reaction mechanism involves a",
        "The pH of a solution is calculated by",
        "Molecular orbital theory explains bonding through",
        "Le Chatelier's principle states that when",
        "The Gibbs free energy determines whether a reaction",
    ],
    "medicine":[
        "The pathophysiology of heart failure involves",
        "Drug-drug interactions in pharmacology occur when",
        "mRNA vaccines work by introducing genetic material",
        "The USMLE tests clinical reasoning about",
        "Acute myocardial infarction presents with",
    ],
}

GENERAL_PROBES=[
    "The capital of France is",
    "Water boils at a temperature of",
    "The speed of light in vacuum is approximately",
    "Photosynthesis converts sunlight into",
    "The Pythagorean theorem states that",
]

def trace_probes(ct_model,probes,batch_size=128,max_features=30):
    from circuit_tracer import attribute
    results=[]
    for p in probes:
        try:
            g=attribute(prompt=p,model=ct_model,max_n_logits=5,
                        batch_size=batch_size,max_feature_nodes=max_features,
                        verbose=False)
            nf=len(g.active_features) if hasattr(g,"active_features") else 0
            adj=g.adjacency_matrix if hasattr(g,"adjacency_matrix") else None
            ne=int(adj.nonzero().shape[0]) if adj is not None else 0
            results.append({"prompt":p,"n_features":nf,"n_edges":ne})
        except Exception as e:
            log.warning("trace failed for '%s': %s",p[:30],e)
            results.append({"prompt":p,"n_features":0,"n_edges":0,"error":str(e)})
    return results

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="/home/ubuntu/circuit_bench")
    ap.add_argument("--anthropic-key",default=os.environ.get("ANTHROPIC_API_KEY",""))
    ap.add_argument("--openai-key",default=os.environ.get("OPENAI_API_KEY",""))
    ap.add_argument("--exa-key",default=os.environ.get("EXA_API_KEY",""))
    ap.add_argument("--hf-token",default=os.environ.get("HF_TOKEN",""))
    ap.add_argument("--claude-model",default="claude-opus-4-6")
    ap.add_argument("--distill-n",type=int,default=50)
    ap.add_argument("--eval-n",type=int,default=50)
    ap.add_argument("--max-steps",type=int,default=150)
    args=ap.parse_args()
    if args.hf_token:os.environ["HF_TOKEN"]=args.hf_token

    od=args.out
    os.makedirs(f"{od}/circuits",exist_ok=True)
    os.makedirs(f"{od}/plots",exist_ok=True)

    log.info("=== CIRCUIT TRACING BENCHMARK: Llama-3.2-1B ===")

    from circuit_tracer import ReplacementModel
    log.info("Loading ReplacementModel (Llama-3.2-1B + llama transcoders)...")
    t0=time.time()
    ct_model=ReplacementModel.from_pretrained("meta-llama/Llama-3.2-1B","llama",
                                                dtype=torch.bfloat16)
    log.info("ReplacementModel loaded in %.1fs",time.time()-t0)

    log.info("=== BASELINE CIRCUIT TRACES ===")
    baseline_traces={}
    for domain,probes in DOMAIN_PROBES.items():
        log.info("Tracing %s domain probes...",domain)
        baseline_traces[domain]=trace_probes(ct_model,probes)
        for r in baseline_traces[domain]:
            log.info("  [%s] features=%d edges=%d '%s'",domain,r["n_features"],r["n_edges"],r["prompt"][:50])
    log.info("Tracing general probes...")
    baseline_traces["general"]=trace_probes(ct_model,GENERAL_PROBES)
    for r in baseline_traces["general"]:
        log.info("  [general] features=%d edges=%d '%s'",r["n_features"],r["n_edges"],r["prompt"][:50])

    with open(f"{od}/circuits/baseline_traces.json","w") as f:
        json.dump(baseline_traces,f,indent=2)
    log.info("Baseline traces saved")

    log.info("=== LOADING TRAINING MODEL ===")
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import get_peft_model,LoraConfig,TaskType
    tok=AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B",trust_remote_code=True)
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    base=AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B",
        torch_dtype=torch.bfloat16,trust_remote_code=True).to("cuda:0")
    lc=LoraConfig(r=32,lora_alpha=64,lora_dropout=0.05,
        target_modules=["q_proj","v_proj"],task_type=TaskType.CAUSAL_LM)
    base=get_peft_model(base,lc)
    base.print_trainable_parameters()

    from wm.bench.eval_harness import DomainEvalHarness
    h=DomainEvalHarness(base,tok,n_samples=100,seed=42)
    log.info("=== BASELINE BENCHMARKS ===")
    bl={}
    for d in ["finance","legal","chemistry","medicine"]:
        bl[d]={"bench":h.eval_domain(d,n=args.eval_n).acc,
               "mmlu":h.eval_mmlu(d,n=args.eval_n).acc}
        log.info("  %s: bench=%.4f mmlu=%.4f",d,bl[d]["bench"],bl[d]["mmlu"])

    from wm.cfg import WMCfg,SearchCfg,GraphCfg,GuardCfg,IterCLCfg,SearchGateCfg,UpdateGateCfg,PaceCfg,DistillCfg,NeurogenesisCfg
    from wm.pipe.loop import AgenticPipeline
    from wm.recipe import NeurogenesisEATRD,NeurogenesisBank
    from wm.bench.iterative import TOPIC_SCHEDULE
    import copy

    cfg=WMCfg(
        search=SearchCfg(exa_api_key=args.exa_key,max_rounds=3,queries_per_round=4,
            extraction_backend="claude" if args.anthropic_key else "regex",
            anthropic_api_key=args.anthropic_key,claude_model=args.claude_model,
            claude_concurrency=8),
        graph=GraphCfg(db_path=f"{od}/graph.db"),
        guard=GuardCfg(enabled=False),
        search_gate=SearchGateCfg(enabled=False),
        update_gate=UpdateGateCfg(enabled=False),
        pace=PaceCfg(web_budget=500,ft_budget=100),
        iter_cl=IterCLCfg(min_steps=50,max_steps=args.max_steps),
        distill=DistillCfg(enabled=bool(args.anthropic_key or args.openai_key),
                            n_questions=args.distill_n,mu_init=1.5,
                            openai_api_key=args.openai_key),
        neurogenesis=NeurogenesisCfg(enabled=True,spawn_loss_thresh=4.0),
    )

    bank=NeurogenesisBank()
    teacher=copy.deepcopy(base).eval()

    report={"baseline_benchmarks":bl,"baseline_traces":baseline_traces,
            "domain_traces":{},"domain_benchmarks":{}}

    DOMAINS=["finance","legal","chemistry","medicine"]
    for di,dom in enumerate(DOMAINS):
        log.info("=== DOMAIN %d/4: %s ===",di+1,dom.upper())
        topics=TOPIC_SCHEDULE.get(dom,[])[:3]

        pipe=AgenticPipeline(cfg,base,tok,domain=dom)
        for ti,topic in enumerate(topics):
            log.info("[%s] Topic %d/%d: %s",dom,ti+1,len(topics),topic)
            try:
                sr=pipe._sg.check(topic,communities=[],model=base,tok=tok,tau=0)
                from wm.search.agent import AgenticSearcher
                searcher=AgenticSearcher(cfg.search,model=base,tok=tok,domain=dom)
                sr_result=searcher.search(topic)
                if sr_result.train_rows:
                    from datasets import Dataset
                    ds=Dataset.from_list(sr_result.train_rows)
                    from wm.graph.dream_gen import gen_dream_prompts
                    dreams=gen_dream_prompts(sr_result.communities,sr_result.claims)
                    r=NeurogenesisEATRD(lr=3e-4,max_steps=args.max_steps,bs=2,temp=2.0,
                        lam_init=0.3,d_targ=0.2,lam_ceil=3.0,mu_init=1.5,max_len=128)
                    tr=r.run(base,teacher,ds,dreams,tok,bank=bank,domain=dom,topic=topic)
                    log.info("  TRAINED: loss=%.4f dream=%.4f steps=%d",tr.loss,tr.dream_loss,tr.steps)
            except Exception as e:
                log.warning("  topic failed: %s",e)

        log.info("=== POST-%s CIRCUIT TRACES ===",dom.upper())
        dom_traces={}
        for probe_dom,probes in DOMAIN_PROBES.items():
            dom_traces[probe_dom]=trace_probes(ct_model,probes)
            for r in dom_traces[probe_dom]:
                bl_feat=baseline_traces[probe_dom][[i for i,b in enumerate(baseline_traces[probe_dom]) if b["prompt"]==r["prompt"]][0]]["n_features"] if any(b["prompt"]==r["prompt"] for b in baseline_traces[probe_dom]) else 0
                delta=r["n_features"]-bl_feat
                log.info("  [%s] features=%d (delta=%+d) '%s'",probe_dom,r["n_features"],delta,r["prompt"][:40])
        dom_traces["general"]=trace_probes(ct_model,GENERAL_PROBES)
        report["domain_traces"][dom]=dom_traces

        post_bench={}
        for d in DOMAINS:
            post_bench[d]={"bench":h.eval_domain(d,n=args.eval_n).acc,
                           "mmlu":h.eval_mmlu(d,n=args.eval_n).acc}
            delta_b=post_bench[d]["bench"]-bl[d]["bench"]
            delta_m=post_bench[d]["mmlu"]-bl[d]["mmlu"]
            log.info("  %s: bench=%.4f(%+.4f) mmlu=%.4f(%+.4f)",d,
                     post_bench[d]["bench"],delta_b,post_bench[d]["mmlu"],delta_m)
        report["domain_benchmarks"][dom]=post_bench

        with open(f"{od}/circuits/traces_after_{dom}.json","w") as f:
            json.dump(dom_traces,f,indent=2)

        try:pipe.close()
        except:pass
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()

    with open(f"{od}/results.json","w") as f:
        json.dump(report,f,indent=2,default=str)
    log.info("=== CIRCUIT TRACING BENCHMARK COMPLETE ===")
    log.info("Results: %s/results.json",od)

if __name__=="__main__":
    main()
