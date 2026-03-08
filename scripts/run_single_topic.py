#!/usr/bin/env python3
import os,sys,copy,time,json,gc,argparse,logging,torch
logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(name)s %(message)s",
                    datefmt="%H:%M:%S",stream=sys.stdout)
os.environ["TOKENIZERS_PARALLELISM"]="false"
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import get_peft_model,PeftModel,LoraConfig,TaskType
from wm.cfg import WMCfg,SearchCfg,GraphCfg,GuardCfg,SearchGateCfg,UpdateGateCfg,PaceCfg,IterCLCfg
from wm.recipe import EATRDRunner
from wm.bench.eval_harness import DomainEvalHarness
from wm.pipe.loop import AgenticPipeline
from wm.eval.anchor import AnchorEval

log=logging.getLogger("single_topic")

def main():
    ap=argparse.ArgumentParser(description="Run a single topic through the full pipeline")
    ap.add_argument("topic",help="topic string e.g. 'credit risk modeling Basel IV'")
    ap.add_argument("--domain",default="finance",choices=["finance","legal","chemistry","medicine"])
    ap.add_argument("--model",default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--adapter",default=None,help="resume from LoRA checkpoint")
    ap.add_argument("--out",default="/tmp/wm_single_topic")
    ap.add_argument("--exa-key",default=os.environ.get("EXA_API_KEY",""))
    ap.add_argument("--anthropic-key",default=os.environ.get("ANTHROPIC_API_KEY",""))
    ap.add_argument("--hf-token",default=os.environ.get("HF_TOKEN",""))
    ap.add_argument("--steps",type=int,default=100)
    ap.add_argument("--lr",type=float,default=2e-4)
    ap.add_argument("--bs",type=int,default=2)
    ap.add_argument("--lora-r",type=int,default=32)
    ap.add_argument("--eval-n",type=int,default=50,help="eval sample size (0=full)")
    ap.add_argument("--save",action="store_true",help="save adapter checkpoint")
    args=ap.parse_args()
    if args.hf_token:os.environ["HF_TOKEN"]=args.hf_token
    os.makedirs(args.out,exist_ok=True)

    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Loading model on %s...",dev)
    tok=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True)
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    base=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,
                                               trust_remote_code=True).to(dev)
    if args.adapter:
        log.info("Loading adapter from %s",args.adapter)
        base=PeftModel.from_pretrained(base,args.adapter).to(dev)
    else:
        lc=LoraConfig(r=args.lora_r,lora_alpha=args.lora_r*2,lora_dropout=0.05,
            target_modules=["q_proj","v_proj"],task_type=TaskType.CAUSAL_LM)
        base=get_peft_model(base,lc)
    base.print_trainable_parameters()

    h=DomainEvalHarness(base,tok,n_samples=100,seed=42)
    anc=AnchorEval(base,tok)

    log.info("PRE-TRAINING EVAL")
    pre_mmlu=h.eval_mmlu(args.domain,n=args.eval_n)
    pre_bench=h.eval_domain(args.domain,n=args.eval_n)
    pre_nll=anc.nll()
    pre_exs=h.generate_examples(args.domain,n=3)
    log.info("  mmlu_%s: %.4f  bench_%s: %.4f  anchor: %.4f",
             args.domain,pre_mmlu.acc,args.domain,pre_bench.acc,pre_nll)
    for ex in pre_exs:
        log.info("  Q: %s",ex["prompt"][:80])
        log.info("  A: %s",ex["response"][:150])

    _last={"result":None}
    def rfn(model,ds,dreams,steps=50,dbank=None):
        teacher=copy.deepcopy(model).eval().to(dev)
        r=EATRDRunner(lr=args.lr,max_steps=steps,bs=args.bs,temp=2.0,
            eps_min=0.01,alpha=0.5,rho=0.01,lam_init=1.0,
            max_len=256,dream_n=4,dream_len=64,
            d_targ=0.5,lam_floor=0.01,lam_ceil=10.0,use_pi=True,
            pi_warmup_frac=0.15)
        res=r.run(model,teacher,ds,dreams,tok,dbank=dbank)
        del teacher;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
        _last["result"]=res
        return res
    rfn.last=_last

    extraction_backend="claude" if args.anthropic_key else "regex"
    cfg=WMCfg(
        search=SearchCfg(exa_api_key=args.exa_key,max_rounds=5,queries_per_round=6,
                         min_claims=5,res_per_query=5,
                         extraction_backend=extraction_backend,
                         anthropic_api_key=args.anthropic_key),
        graph=GraphCfg(db_path=f"{args.out}/graph.db"),
        guard=GuardCfg(enabled=False),
        search_gate=SearchGateCfg(enabled=False),
        update_gate=UpdateGateCfg(enabled=False),
        pace=PaceCfg(web_budget=200,ft_budget=50),
        iter_cl=IterCLCfg(min_steps=args.steps,max_steps=args.steps))

    log.info("RUNNING TOPIC: %s",args.topic)
    t0=time.time()
    pipe=AgenticPipeline(cfg,base,tok,rfn,guard=None)
    pres=pipe.run(args.topic)
    elapsed=time.time()-t0
    log.info("pipe result: %s",{k:v for k,v in pres.items() if k!="pipe"})
    tr=rfn.last["result"]
    if tr:
        log.info("TRAINING: loss=%.4f->%.4f dream=%.4f steps=%d time=%.1fs",
                 tr.history[0]["loss"] if tr.history else 0,
                 tr.history[-1]["loss"] if tr.history else 0,
                 tr.dream_loss or 0,tr.steps,elapsed)
        if tr.history:
            lams=[h_["lambda"] for h_ in tr.history]
            log.info("  lambda: %.4f -> %.4f (mean=%.4f)",lams[0],lams[-1],
                     sum(lams)/len(lams))

    log.info("POST-TRAINING EVAL")
    post_mmlu=h.eval_mmlu(args.domain,n=args.eval_n)
    post_bench=h.eval_domain(args.domain,n=args.eval_n)
    post_nll=anc.nll()
    post_exs=h.generate_examples(args.domain,n=3)
    log.info("  mmlu_%s: %.4f (delta=%+.4f)  bench_%s: %.4f (delta=%+.4f)  anchor: %.4f (delta=%+.4f)",
             args.domain,post_mmlu.acc,post_mmlu.acc-pre_mmlu.acc,
             args.domain,post_bench.acc,post_bench.acc-pre_bench.acc,
             post_nll,post_nll-pre_nll)

    log.info("BEFORE/AFTER COMPARISON:")
    for i in range(min(len(pre_exs),len(post_exs))):
        log.info("  Q: %s",pre_exs[i]["prompt"][:80])
        log.info("  BEFORE: %s",pre_exs[i]["response"][:150])
        log.info("  AFTER:  %s",post_exs[i]["response"][:150])

    if args.save:
        ckp=f"{args.out}/checkpoint"
        os.makedirs(ckp,exist_ok=True)
        base.save_pretrained(ckp)
        log.info("Checkpoint saved to %s",ckp)

    pipe.close()
    log.info("Done in %.1fs",time.time()-t0)

if __name__=="__main__":
    main()
