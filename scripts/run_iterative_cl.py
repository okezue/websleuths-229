#!/usr/bin/env python3
import os,sys,copy,time,json,gc,argparse,logging
logging.basicConfig(level=logging.INFO,format="%(asctime)s %(name)s %(message)s",
                    datefmt="%H:%M:%S",stream=sys.stdout)
os.environ["TOKENIZERS_PARALLELISM"]="false"
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import get_peft_model,LoraConfig,TaskType
from wm.cfg import WMCfg,SearchCfg,GraphCfg,GuardCfg,IterCLCfg,SearchGateCfg,UpdateGateCfg,PaceCfg
from wm.recipe import EATRDRunner,DPMURunner,EABSSCRunner,AdapterBank,SleepConsolidator
from wm.guard.orchestrator import UpdateGuard
from wm.bench.iterative import IterativeCLBench,TOPIC_SCHEDULE
from wm.bench.eval_harness import DomainEvalHarness

DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")
def pr(msg):print(f"\n{'='*70}\n{msg}\n{'='*70}",flush=True)

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

def make_recipe_fn(name,cfg,tok,steps=50,bs=2,lr=2e-4,ml=256,dn=4,dl=64):
    def fn(model,ds,dreams,steps=steps):
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
        del teacher
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
        return res
    return fn

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--recipes",nargs="+",default=["eatrd","dpmu","eab_ssc"])
    ap.add_argument("--steps",type=int,default=50)
    ap.add_argument("--bench-n",type=int,default=100)
    ap.add_argument("--mmlu-n",type=int,default=50)
    ap.add_argument("--lora-r",type=int,default=32)
    ap.add_argument("--ckpt-dir",default="/tmp/wm_checkpoints")
    ap.add_argument("--out",default="/tmp/wm_iterative_cl_results.json")
    ap.add_argument("--exa-key",default=os.environ.get("EXA_API_KEY",""))
    ap.add_argument("--lr",type=float,default=2e-4)
    ap.add_argument("--bs",type=int,default=2)
    ap.add_argument("--ml",type=int,default=256)
    ap.add_argument("--min-steps",type=int,default=30)
    ap.add_argument("--max-steps",type=int,default=150)
    ap.add_argument("--hf-token",default=os.environ.get("HF_TOKEN",""))
    args=ap.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"]=args.hf_token

    pr(f"ITERATIVE CL BENCHMARK — {args.model}")
    print(f"device: {DEV}")
    if DEV.type=="cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    pr(f"LOAD MODEL: {args.model} + LoRA r={args.lora_r}")
    t0=time.time()
    tok=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True)
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    base=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,
                                               trust_remote_code=True)
    base.to(DEV)
    lc=LoraConfig(r=args.lora_r,lora_alpha=args.lora_r*2,lora_dropout=0.05,
        target_modules=["q_proj","v_proj"],task_type=TaskType.CAUSAL_LM)
    base=get_peft_model(base,lc)
    base.print_trainable_parameters()
    snap={k:v.clone() for k,v in base.state_dict().items()}
    print(f"load: {time.time()-t0:.1f}s",flush=True)

    all_reports={}
    cfg=WMCfg(
        search=SearchCfg(exa_api_key=args.exa_key,max_rounds=5,queries_per_round=6,
                         min_claims=5,mmr_lambda=0.7,mmr_k=50,res_per_query=5,
                         round_temps=[0.7,0.9,1.0,1.0,1.0]),
        graph=GraphCfg(db_path="/tmp/wm_iter_cl_graph.db"),
        guard=GuardCfg(enabled=True,max_anchor_delta=0.5),
        search_gate=SearchGateCfg(a=0.4,b=0.3,c=0.3,tau_search=0.1),
        update_gate=UpdateGateCfg(a1=0.3,a2=0.2,a3=0.3,a4=0.2,tau_ready=0.1,tau_novel=0.1),
        pace=PaceCfg(web_budget=100,ft_budget=25,eta=0.05),
        iter_cl=IterCLCfg(bench_n=args.bench_n,mmlu_n=args.mmlu_n,
                          steps_per_topic=args.steps,ckpt_dir=args.ckpt_dir,
                          recipes=args.recipes,min_steps=args.min_steps,
                          max_steps=args.max_steps),
    )

    for rname in args.recipes:
        pr(f"RECIPE: {rname}")
        base.load_state_dict(snap)
        base.to(DEV)
        rfn=make_recipe_fn(rname,cfg,tok,steps=args.steps,bs=args.bs,
                           lr=args.lr,ml=args.ml)
        bench=IterativeCLBench(cfg,base,tok,rfn,rname,
            guard=None,ckpt_dir=args.ckpt_dir,
            bench_n=args.bench_n,mmlu_n=args.mmlu_n)
        report=bench.run()
        print(bench.summary(report))
        rpath=os.path.join(os.path.dirname(args.out),f"wm_iter_{rname}.json")
        bench.save_report(report,rpath)
        print(f"saved: {rpath}")
        all_reports[rname]=report
        base.load_state_dict(snap)
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()

    pr("CROSS-RECIPE COMPARISON")
    comparison={}
    for rname,rpt in all_reports.items():
        fb={d:bs.acc for d,bs in rpt.final_bench.items()}
        fm={d:bs.acc for d,bs in rpt.final_mmlu.items()}
        ga_n=sum(1 for t in rpt.topics if t.guard_accepted)
        comparison[rname]={
            "final_bench":fb,"final_mmlu":fm,
            "final_anchor":rpt.final_anchor,
            "anchor_delta":rpt.final_anchor-rpt.baseline_anchor,
            "guard_accept_rate":ga_n/max(len(rpt.topics),1),
            "total_time":rpt.total_time,
        }
        bench_inst=IterativeCLBench(cfg,base,tok,None,rname,bench_n=args.bench_n)
        mx=bench_inst.retention_matrix(rpt)
        comparison[rname]["retention_matrix"]=mx
    with open(args.out,"w") as f:
        json.dump({"comparison":comparison,
                   "config":{"model":args.model,"lora_r":args.lora_r,
                            "steps":args.steps,"recipes":args.recipes,
                            "bench_n":args.bench_n,"mmlu_n":args.mmlu_n}},
                  f,indent=2,default=str)
    print(f"\nsaved comparison: {args.out}")
    for rname,c in comparison.items():
        print(f"\n{rname}:")
        print(f"  anchor_delta={c['anchor_delta']:+.4f}")
        print(f"  guard_rate={c['guard_accept_rate']:.2f}")
        print(f"  time={c['total_time']:.0f}s")
        for d,a in c["final_bench"].items():
            print(f"  {d}: {a:.4f}")

    pr("ITERATIVE CL BENCHMARK COMPLETE")

if __name__=="__main__":
    main()
