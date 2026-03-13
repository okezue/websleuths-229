#!/usr/bin/env python3
import os,sys,copy,time,json,gc,logging,statistics,torch
logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(name)s %(message)s",
                    datefmt="%H:%M:%S",stream=sys.stdout)
os.environ["TOKENIZERS_PARALLELISM"]="false"
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import get_peft_model,LoraConfig,TaskType
from wm.bench.eval_harness import DomainEvalHarness
from wm.bench.iterative import TOPIC_SCHEDULE
from wm.recipe.neurogenesis import NeurogenesisEATRD,NeurogenesisBank
from wm.cfg import WMCfg,SearchCfg,GraphCfg,GuardCfg,IterCLCfg,SearchGateCfg,UpdateGateCfg,PaceCfg,DistillCfg,NeurogenesisCfg
from wm.pipe.loop import AgenticPipeline
from wm.eval.anchor import AnchorEval
from datasets import Dataset
from wm.graph.dream_gen import gen_dream_prompts,gen_dream_bank
from wm.search.agent import AgenticSearcher
from wm.adapt import adaptive_steps

log=logging.getLogger("deep_train")
DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")
DOMAINS=["finance","legal","chemistry","medicine"]
ALL_BENCHES=["finqa","lexglue","chembench","medqa","gpqa","olympiad","labbench","aime","verus","minif2f"]
PROBES={
    "finance":["The price-to-earnings ratio is calculated by","Credit risk under Basel IV is measured using","A company's balance sheet shows total assets of"],
    "legal":["The Fourth Amendment protects citizens from","Stare decisis means that courts must","The Supreme Court ruled in Miranda v Arizona that"],
    "chemistry":["The SN2 reaction mechanism involves a","The pH of a solution is calculated by","The Gibbs free energy determines whether a reaction"],
    "medicine":["Acute myocardial infarction presents with","Drug-drug interactions in pharmacology occur when","mRNA vaccines work by introducing genetic material"],
    "general":["The capital of France is","Water boils at a temperature of","The Pythagorean theorem states that"],
    "formal_verification":["Write a Verus requires clause for a binary search function","Prove that sorting preserves array length in Lean 4","What loop invariant is needed for a sum accumulator?"],
}

def ts():return time.strftime("%H:%M:%S")

def gen_response(model,tok,prompt,max_tok=150,dev=None):
    if dev is None:dev=next(model.parameters()).device
    model.eval()
    enc=tok(prompt,return_tensors="pt",truncation=True,max_length=256)
    inp={k:v.to(dev) for k,v in enc.items() if k in ("input_ids","attention_mask")}
    with torch.no_grad():
        out=model.generate(**inp,max_new_tokens=max_tok,do_sample=False)
    return tok.decode(out[0][inp["input_ids"].shape[1]:],skip_special_tokens=True)

def full_eval(harness,n=50):
    results={}
    for d in DOMAINS:
        results[f"{d}_bench"]=harness.eval_domain(d,n=n)
        results[f"{d}_mmlu"]=harness.eval_mmlu(d,n=n)
    for b in ALL_BENCHES:
        if b not in [f"{d}_bench" for d in DOMAINS]:
            results[b]=harness.eval_bench(b,n=n)
    return results

def eval_to_dict(results):
    return {k:{"name":v.name,"acc":v.acc,"n":v.n} for k,v in results.items()}

def record_responses(model,tok,dev=None):
    resps={}
    for domain,ps in PROBES.items():
        resps[domain]=[]
        for p in ps:
            r=gen_response(model,tok,p,dev=dev)
            resps[domain].append({"prompt":p,"response":r})
    return resps

def analyze_lora(model):
    from wm.analysis.deep_inspect import analyze_lora_weights
    return analyze_lora_weights(model)

def run_tl_trace(model_name,lora_ckpt,probes_flat,od,tag,dev="cuda:0"):
    try:
        from wm.analysis.tl_circuit import TLCircuitTracer
        tracer=TLCircuitTracer(model_name,lora_ckpt=lora_ckpt,top_k=60,dev=dev)
        results={}
        for domain,p in probes_flat:
            g=tracer.trace(p)
            g.save(f"{od}/{tag}_{domain}.pt")
            tracer.plot(g,f"{od}/{tag}_{domain}.png",title=f"{tag} [{domain}]")
            top3=[]
            if g.influence is not None:
                t=torch.topk(g.influence,min(3,len(g.influence)))
                for i,v in zip(t.indices,t.values):
                    n=g.nodes[i]
                    top3.append({"layer":n.get("layer"),"neuron":n.get("neuron"),
                                 "influence":v.item()})
            results[domain]={"logits":g.logit_targets[:5],"top_neurons":top3}
        del tracer;torch.cuda.empty_cache()
        return results
    except Exception as e:
        log.warning("TL trace failed: %s",e)
        return {}

def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--out",default="/home/ubuntu/deep_train")
    ap.add_argument("--exa-key",default=os.environ.get("EXA_API_KEY",""))
    ap.add_argument("--anthropic-key",default=os.environ.get("ANTHROPIC_API_KEY",""))
    ap.add_argument("--openai-key",default=os.environ.get("OPENAI_API_KEY",""))
    ap.add_argument("--hf-token",default=os.environ.get("HF_TOKEN",""))
    ap.add_argument("--eval-n",type=int,default=50)
    ap.add_argument("--max-steps",type=int,default=200)
    ap.add_argument("--lr",type=float,default=3e-4)
    ap.add_argument("--lora-r",type=int,default=32)
    args=ap.parse_args()
    if args.hf_token:os.environ["HF_TOKEN"]=args.hf_token

    od=args.out
    for d in ["plots","checkpoints","analysis","circuits","responses"]:
        os.makedirs(f"{od}/{d}",exist_ok=True)

    log.info("=== DEEP ANALYSIS TRAINING: %s ===",args.model)
    log.info("loading model...")
    tok=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True)
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    base=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,
        trust_remote_code=True).to(DEV)
    lc=LoraConfig(r=args.lora_r,lora_alpha=args.lora_r*2,lora_dropout=0.05,
        target_modules=["q_proj","v_proj"],task_type=TaskType.CAUSAL_LM)
    base=get_peft_model(base,lc)
    base.print_trainable_parameters()

    teacher=copy.deepcopy(base).eval()
    base_state={n:p.detach().cpu().clone() for n,p in base.named_parameters() if p.requires_grad}
    harness=DomainEvalHarness(base,tok,n_samples=100,seed=42)
    anchor=AnchorEval(base,tok)
    bank=NeurogenesisBank()

    report={"model":args.model,"config":vars(args),"timeline":[]}

    probes_flat=[]
    for domain,ps in PROBES.items():
        probes_flat.append((domain,ps[0]))

    log.info("=== BASELINE EVAL (all 12 benchmarks) ===")
    bl_eval=full_eval(harness,args.eval_n)
    bl_nll=anchor.nll()
    bl_responses=record_responses(base,tok)
    bl_lora=analyze_lora(base)
    log.info("baseline benchmarks:")
    for k,v in bl_eval.items():
        log.info("  %s: acc=%.4f n=%d",k,v.acc,v.n)
    log.info("anchor_nll=%.4f",bl_nll)

    log.info("=== BASELINE CIRCUIT TRACES ===")
    bl_circuits=run_tl_trace(args.model,None,probes_flat,f"{od}/circuits","baseline")

    report["baseline"]={"eval":eval_to_dict(bl_eval),"anchor":bl_nll,
                          "responses":bl_responses,"circuits":bl_circuits}
    with open(f"{od}/responses/baseline.json","w") as f:
        json.dump(bl_responses,f,indent=2)

    cfg=WMCfg(
        search=SearchCfg(exa_api_key=args.exa_key,max_rounds=3,queries_per_round=4,
            extraction_backend="gpt" if args.openai_key else "regex",
            anthropic_api_key=args.anthropic_key,claude_model="gpt-5.4",
            claude_concurrency=8),
        graph=GraphCfg(db_path=f"{od}/graph.db"),
        guard=GuardCfg(enabled=False),
        search_gate=SearchGateCfg(enabled=False),
        update_gate=UpdateGateCfg(enabled=False),
        pace=PaceCfg(web_budget=500,ft_budget=100),
        iter_cl=IterCLCfg(min_steps=50,max_steps=args.max_steps),
        distill=DistillCfg(enabled=bool(args.anthropic_key or args.openai_key),
                            n_questions=50,mu_init=1.5,
                            openai_api_key=args.openai_key,gpt_model="gpt-5.4"),
        neurogenesis=NeurogenesisCfg(enabled=True,spawn_loss_thresh=4.0),
    )

    global_step=0
    for di,dom in enumerate(DOMAINS):
        log.info("\n=== DOMAIN %d/4: %s ===",di+1,dom.upper())
        topics=TOPIC_SCHEDULE.get(dom,[])[:3]

        for ti,topic in enumerate(topics):
            tag=f"{dom}_{ti}"
            log.info("\n[%s] Topic %d/%d: %s",ts(),ti+1,len(topics),topic)

            log.info("  [search] fetching...")
            try:
                searcher=AgenticSearcher(cfg.search,model=base,tok=tok,domain=dom)
                sr=searcher.search(topic)
                log.info("  [search] claims=%d communities=%d",len(sr.claims),len(sr.communities))
            except Exception as e:
                log.warning("  search failed: %s",e);continue

            ds_rows=sr.train_rows if sr.train_rows else [{"text":c.text,"authority":c.confidence} for c in sr.claims[:100]]
            if not ds_rows:
                log.warning("  no training data");continue

            distill_rows=[]
            if cfg.distill.enabled:
                try:
                    from wm.distill.gpt_distill import MultiModelDistill
                    mm=MultiModelDistill(anthropic_key=args.anthropic_key,openai_key=args.openai_key,
                                          concurrency=8,claude_model="gpt-5.4",gpt_model="gpt-5.4")
                    distill_rows=mm.run_sync(dom,50)
                    log.info("  [distill] %d rows",len(distill_rows))
                except Exception as e:
                    log.warning("  distill failed: %s",e)

            ds=Dataset.from_list(ds_rows)
            distill_ds=Dataset.from_list(distill_rows) if distill_rows else None
            dreams=gen_dream_prompts(sr.communities,sr.claims)

            log.info("  [train] %d kg_rows + %d distill_rows + %d dreams",
                     len(ds_rows),len(distill_rows),len(dreams))
            gc.collect();torch.cuda.empty_cache()

            try:
                r=NeurogenesisEATRD(lr=args.lr,max_steps=args.max_steps,bs=2,temp=2.0,
                    lam_init=0.3,d_targ=0.2,lam_ceil=3.0,mu_init=1.5,max_len=128,
                    spawn_loss_thresh=4.0,rank_step=8,max_rank=64)
                tr=r.run(base,teacher,ds,dreams,tok,dbank=None,distill_ds=distill_ds,
                         bank=bank,domain=dom,topic=topic)
                global_step+=tr.steps
                log.info("  [train] loss=%.4f dream=%.4f distill=%.4f steps=%d",
                         tr.loss,tr.dream_loss,tr.extras.get("distill_loss",0),tr.steps)
            except Exception as e:
                log.error("  [train] FAILED: %s",e);continue

            log.info("  [eval] running all benchmarks...")
            post_eval=full_eval(harness,args.eval_n)
            post_nll=anchor.nll()
            for k,v in post_eval.items():
                bl_v=bl_eval.get(k)
                delta=v.acc-(bl_v.acc if bl_v else 0)
                if abs(delta)>0.01:
                    log.info("    %s: %.4f (%+.4f)",k,v.acc,delta)

            log.info("  [responses] recording...")
            post_responses=record_responses(base,tok)
            changed=0
            for domain2,resps in post_responses.items():
                for i,r in enumerate(resps):
                    bl_r=bl_responses.get(domain2,[{}]*10)[min(i,len(bl_responses.get(domain2,[]))-1)]
                    if r["response"][:100]!=bl_r.get("response","")[:100]:
                        changed+=1
                        log.info("    [%s] Q: %s",domain2,r["prompt"][:60])
                        log.info("      BEFORE: %s",bl_r.get("response","")[:80])
                        log.info("      AFTER:  %s",r["response"][:80])
            log.info("  [responses] %d/%d changed",changed,sum(len(v) for v in PROBES.values()))

            log.info("  [lora] analyzing weights...")
            lora_stats=analyze_lora(base)
            total_dw=sum(l.get("dw_norm",0) for l in lora_stats)
            log.info("  [lora] total_dw_norm=%.2f",total_dw)

            log.info("  [circuits] tracing...")
            ckpt_path=f"{od}/checkpoints/{tag}"
            os.makedirs(ckpt_path,exist_ok=True)
            base.save_pretrained(ckpt_path)
            circuits=run_tl_trace(args.model,ckpt_path,probes_flat,
                                   f"{od}/circuits",tag)

            entry={"domain":dom,"topic":topic,"step":global_step,"tag":tag,
                   "train":{"loss":tr.loss,"dream":tr.dream_loss,
                            "distill":tr.extras.get("distill_loss",0),
                            "steps":tr.steps,"n_adapters":tr.extras.get("n_adapters",0)},
                   "eval":eval_to_dict(post_eval),"anchor":post_nll,
                   "responses_changed":changed,"total_dw_norm":total_dw,
                   "circuits":circuits}
            report["timeline"].append(entry)

            with open(f"{od}/responses/{tag}.json","w") as f:
                json.dump(post_responses,f,indent=2)
            with open(f"{od}/report_partial.json","w") as f:
                json.dump(report,f,indent=2,default=str)
            log.info("  [saved] checkpoint + responses + circuits + report")

        gc.collect();torch.cuda.empty_cache()

    log.info("\n=== FINAL EVAL ===")
    final_eval=full_eval(harness,args.eval_n)
    final_nll=anchor.nll()
    final_responses=record_responses(base,tok)

    log.info("\nFINAL RESULTS:")
    log.info("%-25s %10s %10s %10s","Benchmark","Baseline","Final","Delta")
    log.info("-"*57)
    for k in sorted(final_eval.keys()):
        bl_v=bl_eval.get(k)
        fn_v=final_eval[k]
        bl_a=bl_v.acc if bl_v else 0
        delta=fn_v.acc-bl_a
        tag="***" if abs(delta)>0.02 else ""
        log.info("%-25s %10.4f %10.4f %+10.4f %s",k,bl_a,fn_v.acc,delta,tag)

    report["final"]={"eval":eval_to_dict(final_eval),"anchor":final_nll,
                       "responses":final_responses}
    with open(f"{od}/report.json","w") as f:
        json.dump(report,f,indent=2,default=str)
    log.info("\nReport: %s/report.json",od)
    log.info("=== DEEP ANALYSIS TRAINING COMPLETE ===")

if __name__=="__main__":
    main()
