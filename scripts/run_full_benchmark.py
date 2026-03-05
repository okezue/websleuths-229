#!/usr/bin/env python3
import os,sys,copy,time,json,gc,argparse,logging
logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(name)s %(message)s",
                    datefmt="%H:%M:%S",stream=sys.stdout)
os.environ["TOKENIZERS_PARALLELISM"]="false"
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import get_peft_model,LoraConfig,TaskType
from wm.cfg import (WMCfg,SearchCfg,GraphCfg,GuardCfg,IterCLCfg,
                     SearchGateCfg,UpdateGateCfg,PaceCfg)
from wm.recipe import EATRDRunner,DPMURunner,EABSSCRunner
from wm.bench.eval_harness import DomainEvalHarness
from wm.bench.iterative import TOPIC_SCHEDULE
from wm.bench.plots import (plot_train_loss,plot_dream_loss,plot_lambda_evo,
    plot_dual_loss,plot_mmlu_prog,plot_domain_compare,
    plot_retention_heatmap,plot_adaptive_steps)
from wm.pipe.loop import AgenticPipeline
from wm.eval.anchor import AnchorEval

log=logging.getLogger("full_bench")
DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")
DOMAINS=["finance","legal","chemistry","medicine"]

def pr(msg):
    s=f"\n{'='*70}\n{msg}\n{'='*70}"
    print(s,flush=True);log.info(msg)

def make_recipe_fn(name,tok,bs=2,lr=2e-4,ml=256,dn=4,dl=64):
    _last={"result":None}
    def fn(model,ds,dreams,steps=50):
        dev=next(model.parameters()).device
        teacher=copy.deepcopy(model).eval().to(dev)
        res=None
        if name=="eatrd":
            r=EATRDRunner(lr=lr,max_steps=steps,bs=bs,temp=2.0,
                eps_min=0.01,alpha=0.5,rho=0.01,lam_init=1.0,
                max_len=ml,dream_n=dn,dream_len=dl)
            res=r.run(model,teacher,ds,dreams,tok)
        elif name=="dpmu":
            r=DPMURunner(lr=lr,max_steps=steps,bs=bs,temp=2.0,
                n_dream_grads=2,max_len=ml,dream_n=dn,dream_len=dl)
            res=r.run(model,teacher,ds,dreams,tok)
        elif name=="eab_ssc":
            r=EABSSCRunner(lr=lr,max_steps=steps,bs=bs,temp=2.0,
                dream_weight=0.5,max_len=ml,dream_n=dn,dream_len=dl)
            res=r.day(model,teacher,ds,dreams,tok)
        del teacher;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
        _last["result"]=res
        return res
    fn.last=_last
    return fn

def log_bench(tag,scores:dict):
    for k,v in scores.items():
        if hasattr(v,"acc"):
            log.info("  %s %s: acc=%.4f n=%d",tag,k,v.acc,v.n)
        elif isinstance(v,dict) and "acc" in v:
            log.info("  %s %s: acc=%.4f n=%d",tag,k,v["acc"],v.get("n",0))

def bs_dict(bs):
    if hasattr(bs,"acc"):return {"name":bs.name,"acc":bs.acc,"n":bs.n}
    return bs

def main():
    ap=argparse.ArgumentParser(description="Full CL benchmark with backtesting")
    ap.add_argument("--model",default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--recipe",default="eatrd",choices=["eatrd","dpmu","eab_ssc"])
    ap.add_argument("--lora-r",type=int,default=32)
    ap.add_argument("--out",default="/tmp/wm_full_bench")
    ap.add_argument("--exa-key",default=os.environ.get("EXA_API_KEY",""))
    ap.add_argument("--lr",type=float,default=2e-4)
    ap.add_argument("--bs",type=int,default=2)
    ap.add_argument("--ml",type=int,default=256)
    ap.add_argument("--min-steps",type=int,default=30)
    ap.add_argument("--max-steps",type=int,default=150)
    ap.add_argument("--hf-token",default=os.environ.get("HF_TOKEN",""))
    ap.add_argument("--eval-n",type=int,default=0,help="0=full eval, >0=sample N")
    args=ap.parse_args()
    if args.hf_token:os.environ["HF_TOKEN"]=args.hf_token

    od=args.out
    os.makedirs(f"{od}/plots",exist_ok=True)
    os.makedirs(f"{od}/checkpoints",exist_ok=True)
    fh=logging.FileHandler(f"{od}/bench.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s",
                                       datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(fh)

    pr(f"FULL BENCHMARK: {args.model} recipe={args.recipe} lora_r={args.lora_r}")
    log.info("device: %s",DEV)
    if DEV.type=="cuda":
        log.info("  GPU: %s VRAM: %.1fGB",torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_mem/1e9
                 if hasattr(torch.cuda.get_device_properties(0),"total_mem")
                 else torch.cuda.get_device_properties(0).total_memory/1e9)
    log.info("eval_n=%d (%s)",args.eval_n,"FULL" if args.eval_n<=0 else f"sample {args.eval_n}")
    en=args.eval_n

    pr("LOADING MODEL")
    t0=time.time()
    tok=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True)
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    base=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,
                                               trust_remote_code=True).to(DEV)
    lc=LoraConfig(r=args.lora_r,lora_alpha=args.lora_r*2,lora_dropout=0.05,
        target_modules=["q_proj","v_proj"],task_type=TaskType.CAUSAL_LM)
    base=get_peft_model(base,lc)
    base.print_trainable_parameters()
    snap={k:v.clone() for k,v in base.state_dict().items()}
    log.info("model loaded in %.1fs",time.time()-t0)

    rfn=make_recipe_fn(args.recipe,tok,bs=args.bs,lr=args.lr,ml=args.ml)
    report={"config":{"model":args.model,"recipe":args.recipe,"lora_r":args.lora_r,
                       "lr":args.lr,"bs":args.bs,"ml":args.ml,
                       "min_steps":args.min_steps,"max_steps":args.max_steps,
                       "eval_n":en,"device":str(DEV)}}

    pr("PHASE 1: BASELINE FULL EVAL")
    t1=time.time()
    h=DomainEvalHarness(base,tok,n_samples=100,seed=42)
    bl_bench={};bl_mmlu={}
    for d in DOMAINS:
        log.info("  evaluating %s benchmark...",d)
        bl_bench[d]=h.eval_domain(d,n=en)
        log.info("  %s bench: acc=%.4f n=%d",d,bl_bench[d].acc,bl_bench[d].n)
        bl_mmlu[d]=h.eval_mmlu(d,n=en)
        log.info("  %s mmlu:  acc=%.4f n=%d",d,bl_mmlu[d].acc,bl_mmlu[d].n)
    anc=AnchorEval(base,tok)
    bl_nll=anc.nll()
    log.info("  anchor_nll=%.4f",bl_nll)
    log.info("  baseline eval took %.1fs",time.time()-t1)

    bl_exs={}
    for d in DOMAINS:
        bl_exs[d]=h.generate_examples(d,n=3)
        for ex in bl_exs[d]:
            log.info("  [%s] Q: %s",d,ex["prompt"][:60])
            log.info("         A: %s",ex["response"][:120])

    report["baseline"]={
        "bench":{d:bs_dict(bl_bench[d]) for d in DOMAINS},
        "mmlu":{d:bs_dict(bl_mmlu[d]) for d in DOMAINS},
        "anchor":bl_nll,
        "examples":bl_exs,
        "time":time.time()-t1,
    }

    pr("PHASE 2: SEQUENTIAL DOMAIN LEARNING")
    cfg=WMCfg(
        search=SearchCfg(exa_api_key=args.exa_key,max_rounds=5,queries_per_round=6,
                         min_claims=5,mmr_lambda=0.7,mmr_k=50,res_per_query=5,
                         round_temps=[0.7,0.9,1.0,1.0,1.0]),
        graph=GraphCfg(db_path=f"{od}/graph.db"),
        guard=GuardCfg(enabled=False),
        search_gate=SearchGateCfg(a=0.4,b=0.3,c=0.3,tau_search=0.1),
        update_gate=UpdateGateCfg(a1=0.3,a2=0.2,a3=0.3,a4=0.2,tau_ready=0.1,tau_novel=0.1),
        pace=PaceCfg(web_budget=200,ft_budget=50,eta=0.05),
        iter_cl=IterCLCfg(min_steps=args.min_steps,max_steps=args.max_steps),
    )

    mmlu_prog={d:[bl_mmlu[d].acc] for d in DOMAINS}
    ret_mx={d:[] for d in DOMAINS}
    all_topics=[]
    all_asteps=[]
    prev_doms=[]
    topic_idx=0
    report["domains"]={}
    total_ft_time=0.0
    total_ft_steps=0

    for di,dom in enumerate(DOMAINS):
        pr(f"DOMAIN {di+1}/{len(DOMAINS)}: {dom.upper()}")
        topics=TOPIC_SCHEDULE.get(dom,[])
        dom_report={"topics":[],"backtest":{}}

        pipe=AgenticPipeline(cfg,base,tok,rfn,guard=None)
        pipe.pace.state.tau_search=0.1
        pipe.pace.state.tau_ready=0.1

        for ti,topic in enumerate(topics):
            pipe.pace.state.tau_search=0.1
            pipe.pace.state.tau_ready=0.1
            slug=topic.replace(" ","_")[:20]
            log.info("[%d/%d] topic: %s",ti+1,len(topics),topic)

            t_start=time.time()
            try:
                pres=pipe.run(topic)
                log.info("  pipe result: action=%s claims=%s communities=%s asteps=%s",
                         pres.get("action"),pres.get("claims"),
                         pres.get("communities"),pres.get("adaptive_steps"))
            except Exception as e:
                log.warning("  pipe.run failed: %s",e)
                pres={"action":"error","error":str(e)}
            t_topic=time.time()-t_start
            total_ft_time+=t_topic

            tr=rfn.last["result"]
            asteps=pres.get("adaptive_steps",0)
            all_asteps.append(asteps)
            all_topics.append(slug)
            topic_idx+=1

            if tr:
                total_ft_steps+=tr.steps
                log.info("  train: loss=%.4f dream_loss=%.4f steps=%d lr=%.2e time=%.1fs",
                         tr.loss,tr.dream_loss or 0,tr.steps,tr.lr,t_topic)
                if tr.extras:
                    log.info("  extras: %s",{k:f"{v:.4f}" if isinstance(v,float) else v
                                              for k,v in tr.extras.items()})
                if tr.history:
                    log.info("  history: %d points, loss[0]=%.4f loss[-1]=%.4f",
                             len(tr.history),tr.history[0].get("loss",0),
                             tr.history[-1].get("loss",0))
                    if len(tr.history)>1:
                        dl0=tr.history[0].get("dream_loss",0)
                        dlf=tr.history[-1].get("dream_loss",0)
                        log.info("  dream: start=%.4f end=%.4f delta=%+.4f",dl0,dlf,dlf-dl0)
                    pp=f"{od}/plots/{topic_idx:03d}_{slug}"
                    plot_train_loss(tr.history,f"{dom}/{topic}",f"{pp}_loss.png")
                    plot_dream_loss(tr.history,f"{dom}/{topic}",f"{pp}_dream.png")
                    plot_dual_loss(tr.history,f"{dom}/{topic}",f"{pp}_dual.png")
                    if args.recipe=="eatrd":
                        plot_lambda_evo(tr.history,f"{dom}/{topic}",f"{pp}_lambda.png")
                        if tr.history:
                            lam0=tr.history[0].get("lambda",0)
                            lamf=tr.history[-1].get("lambda",0)
                            log.info("  lambda: start=%.4f end=%.4f delta=%+.4f",
                                     lam0,lamf,lamf-lam0)
            else:
                log.info("  no train result (skipped/error)")

            ms=h.eval_mmlu(dom,n=en)
            mmlu_prog[dom].append(ms.acc)
            log.info("  mmlu_%s: acc=%.4f n=%d",dom,ms.acc,ms.n)

            tr_dict={"topic":topic,"domain":dom,"time":t_topic,
                     "adaptive_steps":asteps,"pipe":pres,
                     "mmlu":bs_dict(ms),"topic_idx":topic_idx}
            if tr:
                tr_dict.update({"loss":tr.loss,"dream_loss":tr.dream_loss,
                                "steps":tr.steps,"extras":tr.extras,
                                "history_len":len(tr.history)})
                if tr.history:
                    tr_dict["loss_start"]=tr.history[0].get("loss",0)
                    tr_dict["loss_end"]=tr.history[-1].get("loss",0)
                    tr_dict["dream_start"]=tr.history[0].get("dream_loss",0)
                    tr_dict["dream_end"]=tr.history[-1].get("dream_loss",0)
            dom_report["topics"].append(tr_dict)

        pr(f"DOMAIN {dom.upper()} COMPLETE — FULL EVAL")
        prev_doms.append(dom)

        db=h.eval_domain(dom,n=en)
        log.info("  %s domain bench: acc=%.4f n=%d",dom,db.acc,db.n)
        dom_report["domain_bench"]=bs_dict(db)

        dm={}
        for d2 in DOMAINS:
            ms=h.eval_mmlu(d2,n=en)
            dm[d2]=bs_dict(ms)
            mmlu_prog[d2].append(ms.acc)
            log.info("  mmlu_%s: acc=%.4f n=%d (delta=%+.4f from baseline)",
                     d2,ms.acc,ms.n,ms.acc-bl_mmlu[d2].acc)
        dom_report["mmlu_full"]=dm

        bt={}
        for pd in prev_doms:
            bs=h.eval_domain(pd,n=en)
            bt[pd]=bs_dict(bs)
            delta=bs.acc-bl_bench[pd].acc
            log.info("  backtest %s: acc=%.4f n=%d (delta=%+.4f from baseline)",
                     pd,bs.acc,bs.n,delta)
        dom_report["backtest"]=bt

        for d2 in DOMAINS:
            if d2 in prev_doms:
                ret_mx[d2].append(bt[d2]["acc"])
            else:
                ret_mx[d2].append(0.0)

        cur_nll=anc.nll()
        dom_report["anchor_nll"]=cur_nll
        dom_report["anchor_delta"]=cur_nll-bl_nll
        log.info("  anchor: nll=%.4f delta=%+.4f",cur_nll,cur_nll-bl_nll)

        dom_exs=h.generate_examples(dom,n=3)
        dom_report["examples"]=dom_exs
        log.info("  example responses after %s learning:",dom)
        for ex in dom_exs:
            log.info("    Q: %s",ex["prompt"][:60])
            log.info("    A: %s",ex["response"][:120])

        ckp=f"{od}/checkpoints/{dom}"
        os.makedirs(ckp,exist_ok=True)
        try:
            base.save_pretrained(ckp)
            log.info("  checkpoint saved: %s",ckp)
        except Exception as e:
            log.warning("  checkpoint save failed: %s",e)

        plot_retention_heatmap(ret_mx,steps=prev_doms,
            title=f"Retention after {dom}",path=f"{od}/plots/ret_{dom}.png")
        plot_mmlu_prog(mmlu_prog,title=f"MMLU after {dom}",
            path=f"{od}/plots/mmlu_after_{dom}.png")

        report["domains"][dom]=dom_report

        with open(f"{od}/results_partial.json","w") as f:
            json.dump(report,f,indent=2,default=str)
        log.info("  partial results saved")

        try:pipe.close()
        except:pass
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()

    pr("PHASE 3: FINAL FULL EVAL")
    t3=time.time()
    fn_bench={};fn_mmlu={}
    for d in DOMAINS:
        fn_bench[d]=h.eval_domain(d,n=en)
        fn_mmlu[d]=h.eval_mmlu(d,n=en)
        log.info("  final %s: bench=%.4f mmlu=%.4f",d,fn_bench[d].acc,fn_mmlu[d].acc)
    fn_nll=anc.nll()
    log.info("  final anchor: nll=%.4f delta=%+.4f",fn_nll,fn_nll-bl_nll)

    fn_exs={}
    for d in DOMAINS:
        fn_exs[d]=h.generate_examples(d,n=3)
        log.info("  [%s final] examples:",d)
        for ex in fn_exs[d]:
            log.info("    Q: %s",ex["prompt"][:60])
            log.info("    A: %s",ex["response"][:120])

    report["final"]={
        "bench":{d:bs_dict(fn_bench[d]) for d in DOMAINS},
        "mmlu":{d:bs_dict(fn_mmlu[d]) for d in DOMAINS},
        "anchor":fn_nll,"anchor_delta":fn_nll-bl_nll,
        "examples":fn_exs,
        "time":time.time()-t3,
    }

    pr("PHASE 4: SUMMARY & PLOTS")
    report["summary"]={
        "total_ft_time":total_ft_time,
        "total_ft_steps":total_ft_steps,
        "n_topics":topic_idx,
        "mmlu_prog":mmlu_prog,
        "retention_matrix":ret_mx,
        "adaptive_steps":all_asteps,
        "topics":all_topics,
    }

    plot_mmlu_prog(mmlu_prog,title="MMLU Progression",path=f"{od}/plots/mmlu_prog.png")
    bl_acc={d:bl_bench[d].acc for d in DOMAINS}
    fn_acc={d:fn_bench[d].acc for d in DOMAINS}
    plot_domain_compare(bl_acc,fn_acc,title="Domain Benchmarks: Baseline vs Final",
                        path=f"{od}/plots/domain_cmp.png")
    plot_retention_heatmap(ret_mx,steps=DOMAINS,title="Full Retention Matrix",
                           path=f"{od}/plots/retention_full.png")
    if all_asteps:
        plot_adaptive_steps(all_topics,all_asteps,title="Adaptive Steps per Topic",
                            path=f"{od}/plots/asteps.png")

    bl_mmlu_acc={d:bl_mmlu[d].acc for d in DOMAINS}
    fn_mmlu_acc={d:fn_mmlu[d].acc for d in DOMAINS}
    plot_domain_compare(bl_mmlu_acc,fn_mmlu_acc,title="MMLU: Baseline vs Final",
                        path=f"{od}/plots/mmlu_cmp.png")

    with open(f"{od}/results.json","w") as f:
        json.dump(report,f,indent=2,default=str)

    print(f"\n{'='*70}")
    print(f"FULL BENCHMARK RESULTS: {args.recipe}")
    print(f"{'='*70}")
    print(f"Model: {args.model}  LoRA r={args.lora_r}  LR={args.lr}")
    print(f"Eval mode: {'FULL' if en<=0 else f'sample {en}'}")
    print(f"Total FT time: {total_ft_time:.1f}s  Total FT steps: {total_ft_steps}")
    print(f"Anchor NLL: {bl_nll:.4f} -> {fn_nll:.4f} (delta={fn_nll-bl_nll:+.4f})")
    print(f"\n{'Domain Benchmarks':<20} {'Baseline':>10} {'Final':>10} {'Delta':>10}")
    print("-"*52)
    for d in DOMAINS:
        b=bl_bench[d].acc;f_=fn_bench[d].acc
        print(f"  {d:<18} {b:>10.4f} {f_:>10.4f} {f_-b:>+10.4f}")
    print(f"\n{'MMLU':<20} {'Baseline':>10} {'Final':>10} {'Delta':>10}")
    print("-"*52)
    for d in DOMAINS:
        b=bl_mmlu[d].acc;f_=fn_mmlu[d].acc
        print(f"  {d:<18} {b:>10.4f} {f_:>10.4f} {f_-b:>+10.4f}")
    print(f"\nBacktest retention after final domain:")
    for d in DOMAINS:
        if ret_mx[d]:
            print(f"  {d}: {' -> '.join(f'{v:.4f}' for v in ret_mx[d])}")
    print(f"\nAdaptive steps: min={min(all_asteps) if all_asteps else 0} "
          f"max={max(all_asteps) if all_asteps else 0} "
          f"mean={sum(all_asteps)/max(len(all_asteps),1):.1f}")
    print(f"\nResults: {od}/results.json")
    print(f"Plots:   {od}/plots/")
    print(f"Log:     {od}/bench.log")
    pr("BENCHMARK COMPLETE")

if __name__=="__main__":
    main()
