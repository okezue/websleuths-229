#!/usr/bin/env python3
import os,sys,copy,time,json,gc,argparse,logging,statistics
logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(name)s %(message)s",
                    datefmt="%H:%M:%S",stream=sys.stdout)
os.environ["TOKENIZERS_PARALLELISM"]="false"
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
from peft import get_peft_model,LoraConfig,TaskType
from wm.cfg import (WMCfg,SearchCfg,GraphCfg,GuardCfg,IterCLCfg,
                     SearchGateCfg,UpdateGateCfg,PaceCfg,DistillCfg,NeurogenesisCfg)
from wm.recipe import EATRDRunner,DPMURunner,EABSSCRunner,NeurogenesisEATRD,NeurogenesisBank
from wm.dream.bank import DreamBank
from wm.dream.hdm import hard_dream_mine
from wm.bench.eval_harness import DomainEvalHarness
from wm.bench.iterative import TOPIC_SCHEDULE
from wm.bench.plots import (plot_train_loss,plot_dream_loss,plot_lambda_evo,
    plot_dual_loss,plot_mmlu_prog,plot_domain_compare,
    plot_retention_heatmap,plot_adaptive_steps,
    plot_anchor_drift,plot_dream_bank_growth,plot_before_after)
from wm.pipe.loop import AgenticPipeline
from wm.eval.anchor import AnchorEval

log=logging.getLogger("comprehensive_bench")
DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")
DOMAINS=["finance","legal","chemistry","medicine"]

def pr(msg):
    s=f"\n{'='*70}\n{msg}\n{'='*70}"
    print(s,flush=True);log.info(msg)

def ts():
    return time.strftime("%H:%M:%S")

def make_recipe_fn(name,tok,bs=2,lr=2e-4,ml=256,dn=4,dl=64,
                   mu_init=0.5,mu_floor=0.05,mu_ceil=2.0,
                   neurogenesis=False,ng_cfg=None):
    _last={"result":None}
    _bank={"bank":NeurogenesisBank() if neurogenesis else None}
    def fn(model,ds,dreams,steps=50,dbank=None,distill_ds=None,
           guard_failures=0,domain="",topic=""):
        dev=next(model.parameters()).device
        t0=time.time()
        teacher=copy.deepcopy(model).eval().to(dev)
        log.info("    teacher copy took %.1fs",time.time()-t0)
        res=None
        if name=="eatrd" and neurogenesis:
            nc=ng_cfg or {}
            r=NeurogenesisEATRD(lr=lr,max_steps=steps,bs=bs,temp=2.0,
                eps_min=0.01,alpha=0.5,rho=0.01,lam_init=1.0,
                max_len=ml,dream_n=dn,dream_len=dl,
                d_targ=0.5,lam_floor=0.01,lam_ceil=10.0,use_pi=True,
                pi_warmup_frac=0.15,
                mu_init=mu_init,mu_floor=mu_floor,mu_ceil=mu_ceil,
                spawn_loss_thresh=nc.get("spawn_loss_thresh",2.0),
                spawn_dream_thresh=nc.get("spawn_dream_thresh",1.5),
                spawn_fail_thresh=nc.get("spawn_fail_thresh",2),
                rank_step=nc.get("rank_step",8),
                max_rank=nc.get("max_rank",128),
                ortho_weight=nc.get("ortho_weight",0.01))
            res=r.run(model,teacher,ds,dreams,tok,dbank=dbank,distill_ds=distill_ds,
                      bank=_bank["bank"],guard_failures=guard_failures,
                      domain=domain,topic=topic)
        elif name=="eatrd":
            r=EATRDRunner(lr=lr,max_steps=steps,bs=bs,temp=2.0,
                eps_min=0.01,alpha=0.5,rho=0.01,lam_init=1.0,
                max_len=ml,dream_n=dn,dream_len=dl,
                d_targ=0.5,lam_floor=0.01,lam_ceil=10.0,use_pi=True,
                pi_warmup_frac=0.15,
                mu_init=mu_init,mu_floor=mu_floor,mu_ceil=mu_ceil)
            res=r.run(model,teacher,ds,dreams,tok,dbank=dbank,distill_ds=distill_ds)
        elif name=="dpmu":
            r=DPMURunner(lr=lr,max_steps=steps,bs=bs,temp=2.0,
                n_dream_grads=2,max_len=ml,dream_n=dn,dream_len=dl,
                grad_refresh_k=5,grad_ema_decay=0.9)
            res=r.run(model,teacher,ds,dreams,tok,dbank=dbank)
        elif name=="eab_ssc":
            r=EABSSCRunner(lr=lr,max_steps=steps,bs=bs,temp=2.0,
                dream_weight=0.5,max_len=ml,dream_n=dn,dream_len=dl)
            res=r.day(model,teacher,ds,dreams,tok,dbank=dbank)
        del teacher;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
        if dbank is not None:
            log.info("    DreamBank: %s total=%d",dbank.bucket_sizes(),
                     sum(dbank.bucket_sizes().values()))
        _last["result"]=res
        return res
    fn.last=_last
    fn.bank=_bank
    return fn

def bs_dict(bs):
    if hasattr(bs,"acc"):return {"name":bs.name,"acc":bs.acc,"n":bs.n}
    return bs

def log_recipe_diag(tr,recipe_name,dbank):
    if not tr:return {}
    ex=tr.extras or {}
    diag={}
    if recipe_name=="eatrd":
        diag["lambda"]=ex.get("lambda",0)
        diag["mu"]=ex.get("mu",0)
        diag["eps_k"]=ex.get("eps_k",0)
        diag["d_targ"]=ex.get("d_targ",0)
        diag["use_pi"]=ex.get("use_pi",False)
        diag["distill_loss"]=ex.get("distill_loss",0)
        if tr.history:
            diag["lambda_start"]=tr.history[0].get("lambda",0)
            diag["lambda_end"]=tr.history[-1].get("lambda",0)
            lams=[h.get("lambda",0) for h in tr.history]
            diag["lambda_mean"]=statistics.mean(lams) if lams else 0
            diag["lambda_std"]=statistics.stdev(lams) if len(lams)>1 else 0
            dls=[h.get("distill_loss",0) for h in tr.history]
            diag["distill_loss_mean"]=statistics.mean(dls) if dls else 0
    elif recipe_name=="dpmu":
        if tr.history:
            cached=[h.get("cached",False) for h in tr.history]
            diag["cached_grad_pct"]=sum(cached)/max(len(cached),1)*100
            diag["n_grads"]=tr.history[-1].get("n_grads",0)
    elif recipe_name=="eab_ssc":
        diag["consolidation"]=ex.get("consolidation","unknown")
    if dbank:
        diag["bucket_sizes"]=dbank.bucket_sizes()
        diag["total_prompts"]=sum(dbank.bucket_sizes().values())
    if tr.history:
        losses=[h.get("loss",0) for h in tr.history]
        dlosses=[h.get("dream_loss",0) for h in tr.history]
        diag["loss_start"]=losses[0]
        diag["loss_end"]=losses[-1]
        diag["loss_mean"]=statistics.mean(losses)
        diag["dream_start"]=dlosses[0]
        diag["dream_end"]=dlosses[-1]
        diag["dream_mean"]=statistics.mean(dlosses)
        if len(losses)>=10:
            diag["loss_first10"]=statistics.mean(losses[:10])
            diag["loss_last10"]=statistics.mean(losses[-10:])
            diag["dream_first10"]=statistics.mean(dlosses[:10])
            diag["dream_last10"]=statistics.mean(dlosses[-10:])
    log.info("    recipe_diag: %s",{k:(f"{v:.4f}" if isinstance(v,float) else v) for k,v in diag.items()})
    return diag

def load_model_qlora(model_name,lora_r,dev):
    log.info("loading model: %s",model_name)
    tok=AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    use_qlora=dev.type=="cuda"
    if use_qlora:
        try:
            bnb_cfg=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True)
            base=AutoModelForCausalLM.from_pretrained(
                model_name,quantization_config=bnb_cfg,
                torch_dtype=torch.bfloat16,trust_remote_code=True,
                device_map="auto")
            log.info("loaded with QLoRA 4-bit quantization")
        except Exception as e:
            log.warning("QLoRA load failed (%s), falling back to bf16",e)
            base=AutoModelForCausalLM.from_pretrained(
                model_name,torch_dtype=torch.bfloat16,
                trust_remote_code=True).to(dev)
            use_qlora=False
    else:
        base=AutoModelForCausalLM.from_pretrained(
            model_name,torch_dtype=torch.bfloat16,
            trust_remote_code=True).to(dev)
    try:
        tgt=[]
        for n,_ in base.named_modules():
            if any(k in n for k in ["q_proj","v_proj","k_proj","o_proj",
                                      "gate_proj","up_proj","down_proj"]):
                short=n.split(".")[-1]
                if short not in tgt:tgt.append(short)
        if not tgt:tgt=["q_proj","v_proj"]
    except Exception:
        tgt=["q_proj","v_proj"]
    lc=LoraConfig(r=lora_r,lora_alpha=lora_r*2,lora_dropout=0.05,
        target_modules=tgt,task_type=TaskType.CAUSAL_LM)
    base=get_peft_model(base,lc)
    base.print_trainable_parameters()
    return base,tok,use_qlora

def main():
    ap=argparse.ArgumentParser(description="Comprehensive CL benchmark with procedural KG + distillation")
    ap.add_argument("--model",default="MiniMaxAI/MiniMax-M2.5")
    ap.add_argument("--recipe",default="eatrd",choices=["eatrd","dpmu","eab_ssc"])
    ap.add_argument("--lora-r",type=int,default=64)
    ap.add_argument("--out",default="/tmp/wm_comprehensive_bench")
    ap.add_argument("--exa-key",default=os.environ.get("EXA_API_KEY",""))
    ap.add_argument("--parallel-key",default=os.environ.get("PARALLEL_API_KEY",""))
    ap.add_argument("--anthropic-key",default=os.environ.get("ANTHROPIC_API_KEY",""))
    ap.add_argument("--lr",type=float,default=2e-4)
    ap.add_argument("--bs",type=int,default=2)
    ap.add_argument("--ml",type=int,default=256)
    ap.add_argument("--min-steps",type=int,default=30)
    ap.add_argument("--max-steps",type=int,default=150)
    ap.add_argument("--hf-token",default=os.environ.get("HF_TOKEN",""))
    ap.add_argument("--eval-n",type=int,default=0,help="0=full eval")
    ap.add_argument("--mmlu-every",type=int,default=2)
    ap.add_argument("--example-n",type=int,default=5)
    ap.add_argument("--claude-model",default="claude-sonnet-4-5-20250929")
    ap.add_argument("--claude-concurrency",type=int,default=10)
    ap.add_argument("--distill-n",type=int,default=20)
    ap.add_argument("--mu-init",type=float,default=0.5)
    ap.add_argument("--no-distill",action="store_true")
    ap.add_argument("--no-multi-search",action="store_true")
    ap.add_argument("--no-procedural",action="store_true")
    ap.add_argument("--openai-key",default=os.environ.get("OPENAI_API_KEY",""))
    ap.add_argument("--gpt-model",default="gpt-5.4")
    ap.add_argument("--neurogenesis",action="store_true",default=True)
    ap.add_argument("--no-neurogenesis",action="store_true")
    ap.add_argument("--spawn-loss-thresh",type=float,default=2.0)
    ap.add_argument("--rank-step",type=int,default=8)
    ap.add_argument("--max-rank",type=int,default=128)
    ap.add_argument("--no-claude-thinking",action="store_true")
    args=ap.parse_args()
    if args.hf_token:os.environ["HF_TOKEN"]=args.hf_token

    od=args.out
    os.makedirs(f"{od}/plots",exist_ok=True)
    os.makedirs(f"{od}/checkpoints",exist_ok=True)
    fh=logging.FileHandler(f"{od}/bench.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s",
                                       datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(fh)

    pr(f"COMPREHENSIVE BENCHMARK V3: {args.model} recipe={args.recipe} lora_r={args.lora_r}")
    log.info("device: %s",DEV)
    if DEV.type=="cuda":
        gp=torch.cuda.get_device_properties(0)
        vm=gp.total_mem if hasattr(gp,"total_mem") else gp.total_memory
        log.info("  GPU: %s VRAM: %.1fGB",torch.cuda.get_device_name(0),vm/1e9)
        log.info("  GPU count: %d",torch.cuda.device_count())
    log.info("eval_n=%d (%s)",args.eval_n,"FULL" if args.eval_n<=0 else f"sample {args.eval_n}")
    log.info("extraction: %s","Claude KG + procedural" if args.anthropic_key else "regex (no anthropic key)")
    log.info("search: %s","multi (Exa+Parallel)" if not args.no_multi_search else "Exa only")
    log.info("distill: %s","enabled" if not args.no_distill else "disabled")
    log.info("exa_key: %s  parallel_key: %s","set" if args.exa_key else "NOT SET","set" if args.parallel_key else "NOT SET")
    en=args.eval_n
    exn=args.example_n

    pr("LOADING MODEL")
    t0=time.time()
    base,tok,used_qlora=load_model_qlora(args.model,args.lora_r,DEV)
    teacher_snap=copy.deepcopy(base).eval()
    log.info("model loaded in %.1fs (qlora=%s)",time.time()-t0,used_qlora)

    use_ng=args.neurogenesis and not args.no_neurogenesis
    ng_cfg_dict={"spawn_loss_thresh":args.spawn_loss_thresh,
                  "rank_step":args.rank_step,"max_rank":args.max_rank}
    rfn=make_recipe_fn(args.recipe,tok,bs=args.bs,lr=args.lr,ml=args.ml,
                       mu_init=args.mu_init,neurogenesis=use_ng,ng_cfg=ng_cfg_dict)
    report={"config":{"model":args.model,"recipe":args.recipe,"lora_r":args.lora_r,
                       "lr":args.lr,"bs":args.bs,"ml":args.ml,
                       "min_steps":args.min_steps,"max_steps":args.max_steps,
                       "eval_n":en,"mmlu_every":args.mmlu_every,
                       "example_n":exn,"device":str(DEV),
                       "qlora":used_qlora,
                       "extraction":"claude_kg_procedural" if args.anthropic_key else "regex",
                       "multi_search":not args.no_multi_search,
                       "distill":not args.no_distill,
                       "distill_n":args.distill_n,
                       "mu_init":args.mu_init,
                       "claude_model":args.claude_model if args.anthropic_key else None,
                       "gpt_model":args.gpt_model if args.openai_key else None,
                       "neurogenesis":use_ng,
                       "claude_thinking":not args.no_claude_thinking}}

    pr("PHASE 1: BASELINE FULL EVAL (original model, no fine-tuning)")
    log.info("Testing original model on ALL benchmarks")
    t1=time.time()
    h=DomainEvalHarness(base,tok,n_samples=100,seed=42)
    bl_bench={};bl_mmlu={}
    for d in DOMAINS:
        te=time.time()
        log.info("  [%s] evaluating %s domain benchmark...",ts(),d)
        bl_bench[d]=h.eval_domain(d,n=en)
        log.info("  [%s] %s bench: acc=%.4f n=%d time=%.1fs",ts(),d,bl_bench[d].acc,bl_bench[d].n,time.time()-te)
        te2=time.time()
        log.info("  [%s] evaluating %s MMLU...",ts(),d)
        bl_mmlu[d]=h.eval_mmlu(d,n=en)
        log.info("  [%s] %s MMLU:  acc=%.4f n=%d time=%.1fs",ts(),d,bl_mmlu[d].acc,bl_mmlu[d].n,time.time()-te2)
    log.info("  [%s] evaluating capability benchmarks (GPQA, OlympiadBench, Lab-Bench, AIME)...",ts())
    bl_cap={}
    for cb in ["gpqa","olympiad","labbench","aime"]:
        te=time.time()
        bl_cap[cb]=h.eval_bench(cb,n=en)
        log.info("  [%s] %s: acc=%.4f n=%d time=%.1fs",ts(),cb,bl_cap[cb].acc,bl_cap[cb].n,time.time()-te)
    anc=AnchorEval(base,tok)
    bl_nll=anc.nll()
    log.info("  anchor_nll=%.4f",bl_nll)

    bl_exs={}
    for d in DOMAINS:
        bl_exs[d]=h.generate_examples(d,n=exn)
        for ex in bl_exs[d]:
            log.info("  [%s baseline] Q: %s",d,ex["prompt"][:80])
            log.info("                A: %s",ex["response"][:150])

    ds_sizes={}
    for d in DOMAINS:
        ds_sizes[d]={"bench":bl_bench[d].n,"mmlu":bl_mmlu[d].n}
    t1_elapsed=time.time()-t1
    log.info("  dataset sizes: %s",ds_sizes)
    log.info("  baseline eval took %.1fs (%.1f min)",t1_elapsed,t1_elapsed/60)

    report["baseline"]={
        "bench":{d:bs_dict(bl_bench[d]) for d in DOMAINS},
        "mmlu":{d:bs_dict(bl_mmlu[d]) for d in DOMAINS},
        "capabilities":{cb:bs_dict(bl_cap[cb]) for cb in bl_cap},
        "anchor":bl_nll,
        "examples":bl_exs,
        "dataset_sizes":ds_sizes,
        "time":t1_elapsed,
    }
    with open(f"{od}/results_partial.json","w") as f:
        json.dump(report,f,indent=2,default=str)
    log.info("  baseline results saved")

    pr("PHASE 2: SEQUENTIAL DOMAIN LEARNING WITH PROCEDURAL KG + DISTILLATION")
    log.info("Gates DISABLED — all topics will search + train")
    log.info("Extraction: %s","Claude KG + procedural" if args.anthropic_key else "regex fallback")
    log.info("Multi-search: %s","enabled (Exa+Parallel)" if not args.no_multi_search else "disabled")
    log.info("Distillation: %s","enabled (mu=%.2f)"%args.mu_init if not args.no_distill else "disabled")
    extraction_backend="claude" if args.anthropic_key else "regex"
    cfg=WMCfg(
        search=SearchCfg(
            exa_api_key=args.exa_key,
            parallel_api_key=args.parallel_key,
            max_rounds=5,queries_per_round=6,
            min_claims=5,mmr_lambda=0.7,mmr_k=50,res_per_query=5,
            round_temps=[0.7,0.9,1.0,1.0,1.0],
            extraction_backend=extraction_backend,
            anthropic_api_key=args.anthropic_key,
            claude_model=args.claude_model,
            claude_concurrency=args.claude_concurrency,
            use_multi_search=not args.no_multi_search,
            procedural_extraction=not args.no_procedural),
        graph=GraphCfg(db_path=f"{od}/graph.db"),
        guard=GuardCfg(enabled=False),
        search_gate=SearchGateCfg(enabled=False),
        update_gate=UpdateGateCfg(enabled=False),
        pace=PaceCfg(web_budget=500,ft_budget=100,eta=0.05),
        iter_cl=IterCLCfg(min_steps=args.min_steps,max_steps=args.max_steps),
        distill=DistillCfg(enabled=not args.no_distill,n_questions=args.distill_n,
                            mu_init=args.mu_init,
                            openai_api_key=args.openai_key,
                            gpt_model=args.gpt_model,
                            claude_thinking=not args.no_claude_thinking),
        neurogenesis=NeurogenesisCfg(enabled=use_ng,
                                      spawn_loss_thresh=args.spawn_loss_thresh,
                                      rank_step=args.rank_step,max_rank=args.max_rank),
    )

    mmlu_prog={d:[bl_mmlu[d].acc] for d in DOMAINS}
    ret_mx={d:[] for d in DOMAINS}
    all_topics=[]
    all_asteps=[]
    prev_doms=[]
    topic_idx=0
    report["domains"]={}
    total_ft_time=0.0
    total_search_time=0.0
    total_ft_steps=0
    total_dream_time=0.0
    nll_track=[("baseline",bl_nll)]
    dbank_track=[]
    all_diags=[]
    timing_log=[]
    judge_scores=[]

    for di,dom in enumerate(DOMAINS):
        pr(f"DOMAIN {di+1}/{len(DOMAINS)}: {dom.upper()}")
        topics=TOPIC_SCHEDULE.get(dom,[])
        dom_report={"topics":[],"backtest":{},"diags":[],"timing":{},"judge_scores":[]}
        dom_dbank=None
        dom_ft_time=0.0
        dom_search_time=0.0
        dom_ft_steps=0

        pipe=AgenticPipeline(cfg,base,tok,rfn,guard=None,domain=dom)

        for ti,topic in enumerate(topics):
            slug=topic.replace(" ","_")[:20]
            log.info("")
            log.info("[%s] ===== Topic %d/%d: %s =====",ts(),ti+1,len(topics),topic)

            t_search_start=time.time()
            try:
                pres=pipe.run(topic)
                action=pres.get("action","unknown")
                log.info("  [%s] pipe: action=%s claims=%s communities=%s asteps=%s dreams=%s distill_rows=%s",
                         ts(),action,pres.get("claims"),pres.get("communities"),
                         pres.get("adaptive_steps"),pres.get("dreams"),pres.get("distill_rows",0))
            except Exception as e:
                log.error("  [%s] pipe.run FAILED: %s",ts(),e,exc_info=True)
                pres={"action":"error","error":str(e)}
                action="error"
            t_topic=time.time()-t_search_start

            if action=="train":
                dom_ft_time+=t_topic
                total_ft_time+=t_topic
            dom_search_time+=t_topic
            total_search_time+=t_topic

            tr=rfn.last["result"]
            asteps=pres.get("adaptive_steps",0)
            all_asteps.append(asteps)
            all_topics.append(f"{dom[:3]}_{slug}")
            topic_idx+=1
            dom_dbank=pipe.dbank

            if tr:
                dom_ft_steps+=tr.steps
                total_ft_steps+=tr.steps
                dl_str=f" distill_loss={tr.extras.get('distill_loss',0):.4f}" if tr.extras.get("distill_loss") else ""
                log.info("  [%s] TRAINED: loss=%.4f dream_loss=%.4f%s steps=%d lr=%.2e time=%.1fs",
                         ts(),tr.loss,tr.dream_loss or 0,dl_str,tr.steps,tr.lr,t_topic)
                if tr.history and len(tr.history)>=2:
                    l0=tr.history[0].get("loss",0);lf=tr.history[-1].get("loss",0)
                    d0=tr.history[0].get("dream_loss",0);df=tr.history[-1].get("dream_loss",0)
                    log.info("  loss trajectory: %.4f -> %.4f (delta=%+.4f)",l0,lf,lf-l0)
                    log.info("  dream trajectory: %.4f -> %.4f (delta=%+.4f)",d0,df,df-d0)
                    pp=f"{od}/plots/{topic_idx:03d}_{slug}"
                    plot_train_loss(tr.history,f"{dom}/{topic}",f"{pp}_loss.png")
                    plot_dream_loss(tr.history,f"{dom}/{topic}",f"{pp}_dream.png")
                    plot_dual_loss(tr.history,f"{dom}/{topic}",f"{pp}_dual.png")
                    if args.recipe=="eatrd":
                        plot_lambda_evo(tr.history,f"{dom}/{topic}",f"{pp}_lambda.png")
                diag=log_recipe_diag(tr,args.recipe,dom_dbank)
                if diag:
                    dom_report["diags"].append({"topic":topic,"idx":topic_idx,**diag})
                    all_diags.append({"domain":dom,"topic":topic,"idx":topic_idx,**diag})
            else:
                log.warning("  [%s] NO TRAINING for topic: %s (action=%s)",ts(),topic,action)

            if dom_dbank:
                bsz=dom_dbank.bucket_sizes()
                dbank_track.append({"domain":dom,"topic":topic,"idx":topic_idx,"sizes":bsz})
                log.info("  DreamBank: %s total=%d",bsz,sum(bsz.values()))

            if (ti+1)%args.mmlu_every==0 or ti==len(topics)-1:
                log.info("  [%s] MMLU checkpoint (topic %d)",ts(),ti+1)
                ms=h.eval_mmlu(dom,n=en)
                mmlu_prog[dom].append(ms.acc)
                delta=ms.acc-bl_mmlu[dom].acc
                log.info("  [%s] MMLU_%s: acc=%.4f n=%d (delta=%+.4f vs baseline)",
                         ts(),dom,ms.acc,ms.n,delta)

            if not args.no_distill and args.anthropic_key and (ti+1)%args.mmlu_every==0:
                try:
                    from wm.distill.claude_judge import JudgePipeline
                    jp=JudgePipeline(api_key=args.anthropic_key,
                                     concurrency=args.claude_concurrency,
                                     model=args.claude_model)
                    test_qs=h.generate_examples(dom,n=3)
                    if test_qs:
                        qs_txt=[e["prompt"] for e in test_qs]
                        resp_txt=[e["response"] for e in test_qs]
                        golds=["" for _ in test_qs]
                        jrs=jp.judge_sync(dom,qs_txt,resp_txt,golds)
                        avg_score=statistics.mean(j.overall for j in jrs) if jrs else 0
                        log.info("  [%s] judge scores: avg=%.3f",ts(),avg_score)
                        dom_report["judge_scores"].append(
                            {"topic":topic,"idx":topic_idx,"avg_score":avg_score,
                             "scores":[{"q":j.question[:80],"overall":j.overall,
                                        "correctness":j.correctness} for j in jrs]})
                        judge_scores.append({"domain":dom,"topic":topic,"avg":avg_score})
                except Exception as e:
                    log.warning("  judge eval failed: %s",e)

            tr_dict={"topic":topic,"domain":dom,"time":t_topic,
                     "adaptive_steps":asteps,"pipe":pres,"topic_idx":topic_idx}
            if (ti+1)%args.mmlu_every==0 or ti==len(topics)-1:
                tr_dict["mmlu_checkpoint"]=bs_dict(ms)
            if tr:
                tr_dict.update({"loss":tr.loss,"dream_loss":tr.dream_loss,
                                "distill_loss":tr.extras.get("distill_loss",0),
                                "steps":tr.steps,"extras":tr.extras,
                                "history_len":len(tr.history)})
                if tr.history:
                    tr_dict["loss_start"]=tr.history[0].get("loss",0)
                    tr_dict["loss_end"]=tr.history[-1].get("loss",0)
                    tr_dict["dream_start"]=tr.history[0].get("dream_loss",0)
                    tr_dict["dream_end"]=tr.history[-1].get("dream_loss",0)
                    tr_dict["history"]=tr.history
            dom_report["topics"].append(tr_dict)
            timing_log.append({"domain":dom,"topic":topic,"idx":topic_idx,"time":t_topic,
                               "action":action,"steps":tr.steps if tr else 0})

        pr(f"DOMAIN {dom.upper()} COMPLETE — POST-DOMAIN EVALUATION")
        prev_doms.append(dom)
        dom_report["timing"]={"ft_time":dom_ft_time,"search_time":dom_search_time,
                               "ft_steps":dom_ft_steps}
        log.info("  domain ft_time=%.1fs search_time=%.1fs ft_steps=%d",
                 dom_ft_time,dom_search_time,dom_ft_steps)

        t_eval=time.time()
        log.info("  [%s] evaluating %s domain benchmark...",ts(),dom)
        db=h.eval_domain(dom,n=en)
        bl_d=bl_bench[dom].acc
        log.info("  [%s] %s bench: acc=%.4f n=%d (baseline=%.4f delta=%+.4f)",
                 ts(),dom,db.acc,db.n,bl_d,db.acc-bl_d)
        dom_report["domain_bench"]=bs_dict(db)

        dm={}
        for d2 in DOMAINS:
            log.info("  [%s] evaluating MMLU_%s...",ts(),d2)
            ms=h.eval_mmlu(d2,n=en)
            dm[d2]=bs_dict(ms)
            mmlu_prog[d2].append(ms.acc)
            log.info("  [%s] MMLU_%s: acc=%.4f n=%d (baseline=%.4f delta=%+.4f)",
                     ts(),d2,ms.acc,ms.n,bl_mmlu[d2].acc,ms.acc-bl_mmlu[d2].acc)
        dom_report["mmlu_full"]=dm

        log.info("  [%s] running backtests on previous domains...",ts())
        bt={}
        for pd in prev_doms:
            bs=h.eval_domain(pd,n=en)
            bt[pd]=bs_dict(bs)
            bl_pd=bl_bench[pd].acc
            delta=bs.acc-bl_pd
            tag="IMPROVED" if delta>0.005 else ("DEGRADED" if delta<-0.005 else "STABLE")
            log.info("  [%s] backtest %s: acc=%.4f n=%d (baseline=%.4f delta=%+.4f) [%s]",
                     ts(),pd,bs.acc,bs.n,bl_pd,delta,tag)
        dom_report["backtest"]=bt

        for d2 in DOMAINS:
            if d2 in prev_doms:
                ret_mx[d2].append(bt[d2]["acc"])
            else:
                ret_mx[d2].append(0.0)

        cur_nll=anc.nll()
        dom_report["anchor_nll"]=cur_nll
        dom_report["anchor_delta"]=cur_nll-bl_nll
        nll_track.append((f"after_{dom}",cur_nll))
        nll_tag="STABLE" if abs(cur_nll-bl_nll)<0.1 else ("DRIFT" if cur_nll>bl_nll+0.1 else "BETTER")
        log.info("  [%s] anchor: nll=%.4f delta=%+.4f [%s]",ts(),cur_nll,cur_nll-bl_nll,nll_tag)

        dom_exs=h.generate_examples(dom,n=exn)
        dom_report["examples_after"]=dom_exs
        log.info("  example responses after %s learning:",dom)
        for i,ex in enumerate(dom_exs):
            log.info("    Q%d: %s",i+1,ex["prompt"][:80])
            log.info("    A%d: %s",i+1,ex["response"][:150])
            if i<len(bl_exs.get(dom,[])):
                log.info("    BEFORE: %s",bl_exs[dom][i]["response"][:150])
        plot_before_after(bl_exs.get(dom,[]),dom_exs,dom,
                          f"{od}/plots/before_after_{dom}.png")

        if dom_dbank:
            log.info("  [%s] running hard_dream_mine...",ts())
            try:
                hdm=hard_dream_mine(base,teacher_snap,dom_dbank,tok,
                                    pool_n=50,topk=10,dev=DEV)
                dom_report["hard_dreams"]=[(t,b,float(s)) for t,b,s in hdm]
                log.info("  hard dreams (top-10 KL divergence from teacher):")
                for txt,bk,kl in hdm:
                    log.info("    [%s] kl=%.4f %s",bk,kl,txt[:80])
            except Exception as e:
                log.warning("  hard_dream_mine failed: %s",e)
                dom_report["hard_dreams"]=[]

        dom_report["eval_time"]=time.time()-t_eval

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
        plot_anchor_drift([v for _,v in nll_track],[l for l,_ in nll_track],
            f"{od}/plots/anchor_drift_{dom}.png")
        if dbank_track:
            plot_dream_bank_growth(dbank_track,f"{od}/plots/dbank_growth_{dom}.png")
        if all_asteps:
            plot_adaptive_steps(all_topics,all_asteps,
                title=f"Adaptive Steps (through {dom})",
                path=f"{od}/plots/asteps_{dom}.png")
        bl_acc_so_far={d:bl_bench[d].acc for d in prev_doms}
        fn_acc_so_far={d:bt[d]["acc"] for d in prev_doms}
        plot_domain_compare(bl_acc_so_far,fn_acc_so_far,
            title=f"Domain Bench after {dom}",path=f"{od}/plots/domain_cmp_{dom}.png")

        report["domains"][dom]=dom_report
        with open(f"{od}/results_partial.json","w") as f:
            json.dump(report,f,indent=2,default=str)
        log.info("  partial results saved to %s/results_partial.json",od)

        try:pipe.close()
        except:pass
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()

    pr("PHASE 3: FINAL COMPREHENSIVE EVAL")
    t3=time.time()
    fn_bench={};fn_mmlu={}
    for d in DOMAINS:
        te=time.time()
        log.info("  [%s] final eval: %s benchmark...",ts(),d)
        fn_bench[d]=h.eval_domain(d,n=en)
        log.info("  [%s] final eval: %s MMLU...",ts(),d)
        fn_mmlu[d]=h.eval_mmlu(d,n=en)
        bl_b=bl_bench[d].acc;bl_m=bl_mmlu[d].acc
        log.info("  [%s] final %s: bench=%.4f (bl=%.4f d=%+.4f) mmlu=%.4f (bl=%.4f d=%+.4f) time=%.1fs",
                 ts(),d,fn_bench[d].acc,bl_b,fn_bench[d].acc-bl_b,
                 fn_mmlu[d].acc,bl_m,fn_mmlu[d].acc-bl_m,time.time()-te)
    fn_cap={}
    for cb in ["gpqa","olympiad","labbench","aime"]:
        te=time.time()
        fn_cap[cb]=h.eval_bench(cb,n=en)
        bl_c=bl_cap[cb].acc
        log.info("  [%s] final %s: acc=%.4f (bl=%.4f d=%+.4f) time=%.1fs",
                 ts(),cb,fn_cap[cb].acc,bl_c,fn_cap[cb].acc-bl_c,time.time()-te)
    fn_nll=anc.nll()
    nll_track.append(("final",fn_nll))
    log.info("  final anchor: nll=%.4f delta=%+.4f",fn_nll,fn_nll-bl_nll)

    fn_exs={}
    for d in DOMAINS:
        fn_exs[d]=h.generate_examples(d,n=exn)
        log.info("  [%s final] %s examples:",d.upper(),d)
        for i,ex in enumerate(fn_exs[d]):
            log.info("    Q%d: %s",i+1,ex["prompt"][:80])
            log.info("    BEFORE: %s",bl_exs.get(d,[{}]*exn)[min(i,len(bl_exs.get(d,[]))-1)].get("response","")[:120] if bl_exs.get(d) else "N/A")
            log.info("    AFTER:  %s",ex["response"][:120])
        plot_before_after(bl_exs.get(d,[]),fn_exs[d],d,f"{od}/plots/before_after_final_{d}.png")

    report["final"]={
        "bench":{d:bs_dict(fn_bench[d]) for d in DOMAINS},
        "mmlu":{d:bs_dict(fn_mmlu[d]) for d in DOMAINS},
        "capabilities":{cb:bs_dict(fn_cap[cb]) for cb in fn_cap},
        "anchor":fn_nll,"anchor_delta":fn_nll-bl_nll,
        "examples":fn_exs,
        "time":time.time()-t3,
    }

    pr("PHASE 4: ANALYSIS & SUMMARY")
    report["summary"]={
        "total_ft_time":total_ft_time,
        "total_search_time":total_search_time,
        "total_ft_steps":total_ft_steps,
        "n_topics":topic_idx,
        "mmlu_prog":mmlu_prog,
        "retention_matrix":ret_mx,
        "adaptive_steps":all_asteps,
        "topics":all_topics,
        "anchor_track":nll_track,
        "dbank_growth":dbank_track,
        "recipe_diags":all_diags,
        "timing_log":timing_log,
        "judge_scores":judge_scores,
    }

    plot_mmlu_prog(mmlu_prog,title="MMLU Progression (all domains)",path=f"{od}/plots/mmlu_prog.png")
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
    plot_anchor_drift([v for _,v in nll_track],[l for l,_ in nll_track],
        f"{od}/plots/anchor_drift_full.png")
    if dbank_track:
        plot_dream_bank_growth(dbank_track,f"{od}/plots/dbank_growth_full.png")

    with open(f"{od}/results.json","w") as f:
        json.dump(report,f,indent=2,default=str)

    pr("FINAL RESULTS SUMMARY")
    print(f"Model: {args.model}  LoRA r={args.lora_r}  LR={args.lr}  Recipe={args.recipe}")
    print(f"QLoRA: {used_qlora}  Multi-search: {not args.no_multi_search}  Distill: {not args.no_distill}  Neurogenesis: {use_ng}")
    print(f"GPT model: {args.gpt_model if args.openai_key else 'none'}  Claude thinking: {not args.no_claude_thinking}")
    print(f"Extraction: {'Claude KG + procedural' if args.anthropic_key else 'regex'}")
    print(f"Eval mode: {'FULL' if en<=0 else f'sample {en}'}  MMLU every {args.mmlu_every} topics")
    print(f"Total FT time: {total_ft_time:.1f}s ({total_ft_time/60:.1f}min)")
    print(f"Total search time: {total_search_time:.1f}s ({total_search_time/60:.1f}min)")
    print(f"Total FT steps: {total_ft_steps}")
    print(f"Anchor NLL: {bl_nll:.4f} -> {fn_nll:.4f} (delta={fn_nll-bl_nll:+.4f})")

    print(f"\n{'Domain Benchmarks':<22} {'Baseline':>10} {'Final':>10} {'Delta':>10} {'N':>8}")
    print("-"*62)
    for d in DOMAINS:
        b=bl_bench[d].acc;f_=fn_bench[d].acc
        print(f"  {d:<20} {b:>10.4f} {f_:>10.4f} {f_-b:>+10.4f} {fn_bench[d].n:>8}")

    print(f"\n{'MMLU':<22} {'Baseline':>10} {'Final':>10} {'Delta':>10} {'N':>8}")
    print("-"*62)
    for d in DOMAINS:
        b=bl_mmlu[d].acc;f_=fn_mmlu[d].acc
        print(f"  {d:<20} {b:>10.4f} {f_:>10.4f} {f_-b:>+10.4f} {fn_mmlu[d].n:>8}")

    print(f"\n{'Capabilities':<22} {'Baseline':>10} {'Final':>10} {'Delta':>10} {'N':>8}")
    print("-"*62)
    for cb in ["gpqa","olympiad","labbench","aime"]:
        b=bl_cap[cb].acc;f_=fn_cap[cb].acc
        print(f"  {cb:<20} {b:>10.4f} {f_:>10.4f} {f_-b:>+10.4f} {fn_cap[cb].n:>8}")

    print(f"\nBacktest retention after each domain:")
    for d in DOMAINS:
        if ret_mx[d]:
            vals=ret_mx[d]
            print(f"  {d}: {' -> '.join(f'{v:.4f}' for v in vals)}")

    print(f"\nAnchor NLL progression:")
    for label,val in nll_track:
        delta=val-bl_nll
        print(f"  {label:<20} {val:.4f} ({delta:+.4f})")

    if judge_scores:
        print(f"\nJudge scores by domain:")
        for dom in DOMAINS:
            ds=[j["avg"] for j in judge_scores if j["domain"]==dom]
            if ds:
                print(f"  {dom}: avg={statistics.mean(ds):.3f} n={len(ds)}")

    if dbank_track:
        last_db=dbank_track[-1]["sizes"]
        print(f"\nFinal DreamBank: {sum(last_db.values())} total prompts")
        for bk,cnt in last_db.items():
            print(f"  {bk}: {cnt}")

    if all_asteps:
        print(f"\nAdaptive steps: min={min(all_asteps)} max={max(all_asteps)} "
              f"mean={statistics.mean(all_asteps):.1f} "
              f"median={statistics.median(all_asteps):.1f}")

    print(f"\nTiming breakdown:")
    for dom in DOMAINS:
        dd=report.get("domains",{}).get(dom,{}).get("timing",{})
        if dd:
            print(f"  {dom}: ft={dd.get('ft_time',0):.1f}s steps={dd.get('ft_steps',0)} "
                  f"search={dd.get('search_time',0):.1f}s")

    print(f"\nDataset sizes (full eval):")
    for d,sz in ds_sizes.items():
        print(f"  {d}: bench={sz['bench']} mmlu={sz['mmlu']}")

    if all_diags:
        print(f"\nRecipe diagnostics summary ({args.recipe}):")
        if args.recipe=="eatrd":
            lams=[d.get("lambda",0) for d in all_diags if "lambda" in d]
            if lams:
                print(f"  lambda: mean={statistics.mean(lams):.4f} "
                      f"std={statistics.stdev(lams) if len(lams)>1 else 0:.4f}")
            dls=[d.get("distill_loss",0) for d in all_diags if "distill_loss" in d]
            if dls:
                print(f"  distill_loss: mean={statistics.mean(dls):.4f}")
        elif args.recipe=="dpmu":
            cpcts=[d.get("cached_grad_pct",0) for d in all_diags if "cached_grad_pct" in d]
            if cpcts:
                print(f"  cached_grad_pct: mean={statistics.mean(cpcts):.1f}%")

    print(f"\nResults: {od}/results.json")
    print(f"Plots:   {od}/plots/")
    print(f"Log:     {od}/bench.log")
    pr("COMPREHENSIVE BENCHMARK V3 COMPLETE")

if __name__=="__main__":
    main()
