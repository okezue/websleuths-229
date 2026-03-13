#!/usr/bin/env python3
import os,sys,copy,time,json,gc,logging,argparse,torch
logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(name)s %(message)s",
                    datefmt="%H:%M:%S",stream=sys.stdout)
os.environ["TOKENIZERS_PARALLELISM"]="false"
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import get_peft_model,LoraConfig,TaskType
from wm.bench.eval_harness import DomainEvalHarness
from wm.bench.iterative import TOPIC_SCHEDULE
from wm.recipe.neurogenesis import NeurogenesisEATRD,NeurogenesisBank
from wm.cfg import WMCfg,SearchCfg,GraphCfg,GuardCfg,IterCLCfg,DistillCfg,NeurogenesisCfg,SearchGateCfg,UpdateGateCfg,PaceCfg
from wm.pipe.loop import AgenticPipeline
from wm.eval.anchor import AnchorEval
from wm.search.agent import AgenticSearcher
from wm.gate.authority import AUTHORITY_FUNCS
from wm.adapt import adaptive_steps
from datasets import Dataset
from wm.graph.dream_gen import gen_dream_prompts,gen_dream_bank

log=logging.getLogger("authority_sweep")
DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")
DOMAINS=["finance","legal","chemistry","medicine"]
ALL_BENCHES=["finqa","lexglue","chembench","medqa","gpqa","olympiad","labbench","aime","verus","minif2f","multikernelbench"]
PROBES={
    "finance":["The price-to-earnings ratio is calculated by","Credit risk under Basel IV is measured using","A company's balance sheet shows total assets of"],
    "legal":["The Fourth Amendment protects citizens from","Stare decisis means that courts must","The Supreme Court ruled in Miranda v Arizona that"],
    "chemistry":["The SN2 reaction mechanism involves a","The pH of a solution is calculated by","The Gibbs free energy determines whether a reaction"],
    "medicine":["Acute myocardial infarction presents with","Drug-drug interactions in pharmacology occur when","mRNA vaccines work by introducing genetic material"],
    "general":["The capital of France is","Water boils at a temperature of","The Pythagorean theorem states that"],
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
    r={}
    for d in DOMAINS:
        r[f"{d}_bench"]=harness.eval_domain(d,n=n)
        r[f"{d}_mmlu"]=harness.eval_mmlu(d,n=n)
    for b in ALL_BENCHES:
        if b not in [f"{d}_bench" for d in DOMAINS]:
            r[b]=harness.eval_bench(b,n=n)
    return r

def eval_to_dict(results):
    d={}
    for k,v in results.items():
        d[k]={"name":v.name,"acc":v.acc,"n":v.n}
        if v.extras:d[k]["extras"]=v.extras
    return d

def record_responses(model,tok,dev=None):
    resps={}
    for domain,ps in PROBES.items():
        resps[domain]=[]
        for p in ps:
            r=gen_response(model,tok,p,dev=dev)
            resps[domain].append({"prompt":p,"response":r})
    return resps

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

def analyze_lora(model):
    try:
        from wm.analysis.deep_inspect import analyze_lora_weights
        return analyze_lora_weights(model)
    except Exception:
        return []

def run_single_authority(args,auth_name,base_state,model_name):
    log.info("\n{'='*60}")
    log.info("=== AUTHORITY RUN: %s ===",auth_name.upper())
    log.info("{'='*60}")

    od=f"{args.out}/{auth_name}"
    for d in ["plots","checkpoints","analysis","circuits","responses"]:
        os.makedirs(f"{od}/{d}",exist_ok=True)

    tok=AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.bfloat16,
        trust_remote_code=True).to(DEV)
    lc=LoraConfig(r=args.lora_r,lora_alpha=args.lora_r*2,lora_dropout=0.05,
        target_modules=["q_proj","v_proj"],task_type=TaskType.CAUSAL_LM)
    model=get_peft_model(model,lc)
    model.print_trainable_parameters()

    teacher=copy.deepcopy(model).eval()
    harness=DomainEvalHarness(model,tok,n_samples=100,seed=42)
    anchor=AnchorEval(model,tok)
    bank=NeurogenesisBank()

    probes_flat=[(d,ps[0]) for d,ps in PROBES.items()]

    log.info("[%s] baseline eval...",auth_name)
    bl_eval=full_eval(harness,args.eval_n)
    bl_nll=anchor.nll()
    bl_responses=record_responses(model,tok)

    report={"authority":auth_name,"model":model_name,"config":vars(args),
            "baseline":{"eval":eval_to_dict(bl_eval),"anchor":bl_nll},
            "timeline":[]}

    os.environ["WM_AUTHORITY"]=auth_name
    cfg=WMCfg(
        search=SearchCfg(authority_func=auth_name,
            exa_api_key=args.exa_key,max_rounds=3,queries_per_round=4,
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
        log.info("\n[%s] DOMAIN %d/4: %s",auth_name,di+1,dom.upper())
        topics=TOPIC_SCHEDULE.get(dom,[])[:3]

        for ti,topic in enumerate(topics):
            tag=f"{dom}_{ti}"
            log.info("[%s][%s] Topic %d/%d: %s",ts(),auth_name,ti+1,len(topics),topic)

            try:
                searcher=AgenticSearcher(cfg.search,model=model,tok=tok,domain=dom)
                sr=searcher.search(topic)
                log.info("  claims=%d communities=%d",len(sr.claims),len(sr.communities))
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
                except Exception as e:
                    log.warning("  distill failed: %s",e)

            ds=Dataset.from_list(ds_rows)
            distill_ds=Dataset.from_list(distill_rows) if distill_rows else None
            dreams=gen_dream_prompts(sr.communities,sr.claims)

            log.info("  %d kg + %d distill + %d dreams",len(ds_rows),len(distill_rows),len(dreams))
            gc.collect();torch.cuda.empty_cache()

            try:
                r=NeurogenesisEATRD(lr=args.lr,max_steps=args.max_steps,bs=2,temp=2.0,
                    lam_init=0.3,d_targ=0.2,lam_ceil=3.0,mu_init=1.5,max_len=128,
                    spawn_loss_thresh=4.0,rank_step=8,max_rank=64)
                tr=r.run(model,teacher,ds,dreams,tok,dbank=None,distill_ds=distill_ds,
                         bank=bank,domain=dom,topic=topic)
                global_step+=tr.steps
                log.info("  loss=%.4f dream=%.4f distill=%.4f steps=%d",
                         tr.loss,tr.dream_loss,tr.extras.get("distill_loss",0),tr.steps)
            except Exception as e:
                log.error("  train FAILED: %s",e);continue

            post_eval=full_eval(harness,args.eval_n)
            post_nll=anchor.nll()
            post_responses=record_responses(model,tok)
            lora_stats=analyze_lora(model)

            ckpt_path=f"{od}/checkpoints/{tag}"
            os.makedirs(ckpt_path,exist_ok=True)
            model.save_pretrained(ckpt_path)
            circuits=run_tl_trace(model_name,ckpt_path,probes_flat,
                                   f"{od}/circuits",tag)

            changed=0
            for d2,resps in post_responses.items():
                for i,rv in enumerate(resps):
                    bl_r=bl_responses.get(d2,[{}]*10)[min(i,len(bl_responses.get(d2,[]))-1)]
                    if rv["response"][:100]!=bl_r.get("response","")[:100]:changed+=1

            entry={"domain":dom,"topic":topic,"step":global_step,"tag":tag,
                   "train":{"loss":tr.loss,"dream":tr.dream_loss,
                            "distill":tr.extras.get("distill_loss",0),
                            "steps":tr.steps,"n_adapters":tr.extras.get("n_adapters",0)},
                   "eval":eval_to_dict(post_eval),"anchor":post_nll,
                   "responses_changed":changed,
                   "total_dw_norm":sum(l.get("dw_norm",0) for l in lora_stats),
                   "circuits":circuits}
            report["timeline"].append(entry)

            with open(f"{od}/responses/{tag}.json","w") as f:
                json.dump(post_responses,f,indent=2)
            with open(f"{od}/report_partial.json","w") as f:
                json.dump(report,f,indent=2,default=str)

        gc.collect();torch.cuda.empty_cache()

    log.info("\n[%s] FINAL EVAL",auth_name)
    final_eval=full_eval(harness,args.eval_n)
    final_nll=anchor.nll()
    report["final"]={"eval":eval_to_dict(final_eval),"anchor":final_nll}

    with open(f"{od}/report.json","w") as f:
        json.dump(report,f,indent=2,default=str)

    del model,teacher,harness
    gc.collect();torch.cuda.empty_cache()
    return report

def run_rl_phase(args,best_auth,model_name):
    log.info("\n{'='*60}")
    log.info("=== RL PHASE: %s (best authority) ===",best_auth.upper())
    log.info("{'='*60}")

    od=f"{args.out}/rl_{best_auth}"
    for d in ["checkpoints","responses","circuits"]:
        os.makedirs(f"{od}/{d}",exist_ok=True)

    best_ckpt=f"{args.out}/{best_auth}/checkpoints"
    last_ckpt=None
    for dom in reversed(DOMAINS):
        for ti in range(5,-1,-1):
            p=f"{best_ckpt}/{dom}_{ti}"
            if os.path.isdir(p):last_ckpt=p;break
        if last_ckpt:break

    tok=AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.bfloat16,
        trust_remote_code=True).to(DEV)
    lc=LoraConfig(r=args.lora_r,lora_alpha=args.lora_r*2,lora_dropout=0.05,
        target_modules=["q_proj","v_proj"],task_type=TaskType.CAUSAL_LM)
    model=get_peft_model(model,lc)

    if last_ckpt:
        from peft import PeftModel
        log.info("loading best checkpoint: %s",last_ckpt)
        try:
            model.load_adapter(last_ckpt,adapter_name="default")
        except Exception as e:
            log.warning("adapter load failed: %s, starting fresh",e)

    teacher=copy.deepcopy(model).eval()
    harness=DomainEvalHarness(model,tok,n_samples=100,seed=42)

    bl_eval=full_eval(harness,args.eval_n)
    report={"phase":"rl","authority":best_auth,"baseline":eval_to_dict(bl_eval),"rounds":[]}

    ok=args.openai_key
    if not ok:
        log.warning("no openai key, skipping RL (needs GPT-as-judge)")
        report["skipped"]="no_openai_key"
        with open(f"{od}/report.json","w") as f:
            json.dump(report,f,indent=2,default=str)
        return report

    from wm.distill.self_play import SelfPlayDistill
    sp=SelfPlayDistill(openai_key=ok,anthropic_key=args.anthropic_key,
                       gpt_model="gpt-5.4",concurrency=8)

    lam_vals=[0.1,0.2,0.3,0.5,0.7,1.0]
    mu_vals=[0.5,1.0,1.5,2.0,3.0]

    best_score=0
    best_lam=0.3
    best_mu=1.5
    for ri,(lam,mu) in enumerate([(l,m) for l in lam_vals[:3] for m in mu_vals[:2]]):
        log.info("[RL round %d] lam=%.2f mu=%.2f",ri+1,lam,mu)

        rl_rows=[]
        for dom in DOMAINS:
            try:
                rows=sp.run_sync(dom,n_problems=20,n_harder=10,judge_thresh=0.6)
                rl_rows.extend(rows)
            except Exception as e:
                log.warning("self-play %s failed: %s",dom,e)

        if not rl_rows:
            log.warning("no RL data, skipping round");continue

        ood_rows=sp.gen_ood_negative(10)
        rl_rows.extend(ood_rows)

        ds=Dataset.from_list(rl_rows)
        dreams=[]

        try:
            from wm.recipe.neurogenesis import NeurogenesisEATRD
            nr=NeurogenesisEATRD(lr=args.lr*0.5,max_steps=min(args.max_steps,100),bs=2,
                temp=2.0,lam_init=lam,d_targ=0.2,lam_ceil=3.0,mu_init=mu,max_len=128,
                spawn_loss_thresh=4.0,rank_step=8,max_rank=64)
            tr=nr.run(model,teacher,ds,dreams,tok)
            log.info("  loss=%.4f dream=%.4f steps=%d",tr.loss,tr.dream_loss,tr.steps)
        except Exception as e:
            log.error("  RL train failed: %s",e);continue

        ev=full_eval(harness,args.eval_n)
        avg=sum(v.acc for v in ev.values())/max(len(ev),1)
        log.info("  avg_acc=%.4f",avg)

        if avg>best_score:
            best_score=avg;best_lam=lam;best_mu=mu
            ckpt=f"{od}/checkpoints/rl_best"
            os.makedirs(ckpt,exist_ok=True)
            model.save_pretrained(ckpt)

        report["rounds"].append({"lam":lam,"mu":mu,"avg_acc":avg,
                                  "eval":eval_to_dict(ev),"loss":tr.loss})

    report["best"]={"lam":best_lam,"mu":best_mu,"avg_acc":best_score}

    probes_flat=[(d,ps[0]) for d,ps in PROBES.items()]
    rl_ckpt=f"{od}/checkpoints/rl_best"
    if os.path.isdir(rl_ckpt):
        circuits=run_tl_trace(model_name,rl_ckpt,probes_flat,f"{od}/circuits","rl_best")
        report["circuits"]=circuits

    with open(f"{od}/report.json","w") as f:
        json.dump(report,f,indent=2,default=str)

    del model,teacher,harness
    gc.collect();torch.cuda.empty_cache()
    return report

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--out",default="/home/ubuntu/authority_sweep")
    ap.add_argument("--exa-key",default=os.environ.get("EXA_API_KEY",""))
    ap.add_argument("--anthropic-key",default=os.environ.get("ANTHROPIC_API_KEY",""))
    ap.add_argument("--openai-key",default=os.environ.get("OPENAI_API_KEY",""))
    ap.add_argument("--hf-token",default=os.environ.get("HF_TOKEN",""))
    ap.add_argument("--eval-n",type=int,default=50)
    ap.add_argument("--max-steps",type=int,default=200)
    ap.add_argument("--lr",type=float,default=3e-4)
    ap.add_argument("--lora-r",type=int,default=32)
    ap.add_argument("--authorities",nargs="+",default=list(AUTHORITY_FUNCS.keys()))
    ap.add_argument("--skip-rl",action="store_true")
    ap.add_argument("--only-rl",default="")
    args=ap.parse_args()
    if args.hf_token:os.environ["HF_TOKEN"]=args.hf_token

    os.makedirs(args.out,exist_ok=True)
    log.info("=== AUTHORITY SWEEP: %s ===",args.model)
    log.info("authorities: %s",args.authorities)
    log.info("output: %s",args.out)

    if args.only_rl:
        report=run_rl_phase(args,args.only_rl,args.model)
        log.info("RL report: %s/rl_%s/report.json",args.out,args.only_rl)
        return

    all_reports={}
    base_state=None

    for auth in args.authorities:
        t0=time.time()
        report=run_single_authority(args,auth,base_state,args.model)
        dt=time.time()-t0
        all_reports[auth]=report
        log.info("[%s] completed in %.1f min",auth,dt/60)

    log.info("\n=== COMPARISON SUMMARY ===")
    log.info("%-15s %10s %10s %10s %10s %10s","Authority","FinQA","LexGLUE","ChemBench","MedQA","Avg")
    log.info("-"*67)
    best_auth=None
    best_avg=0
    for auth,rep in all_reports.items():
        fe=rep.get("final",{}).get("eval",{})
        scores=[]
        for k in ["finance_bench","legal_bench","chemistry_bench","medicine_bench"]:
            s=fe.get(k,{}).get("acc",0)
            scores.append(s)
        avg=sum(scores)/max(len(scores),1)
        log.info("%-15s %10.4f %10.4f %10.4f %10.4f %10.4f",
                 auth,*scores,avg)
        if avg>best_avg:best_avg=avg;best_auth=auth

    log.info("\nBest authority: %s (avg=%.4f)",best_auth,best_avg)

    with open(f"{args.out}/sweep_summary.json","w") as f:
        json.dump({"authorities":list(all_reports.keys()),
                    "best":best_auth,"best_avg":best_avg,
                    "reports":{k:v.get("final",{}) for k,v in all_reports.items()}},
                  f,indent=2,default=str)

    if not args.skip_rl and best_auth:
        rl_report=run_rl_phase(args,best_auth,args.model)
        all_reports["rl"]=rl_report

    with open(f"{args.out}/full_sweep.json","w") as f:
        json.dump(all_reports,f,indent=2,default=str)

    log.info("\n=== AUTHORITY SWEEP COMPLETE ===")
    log.info("Reports: %s/",args.out)

if __name__=="__main__":
    main()
