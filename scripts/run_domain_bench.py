#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,logging
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import get_peft_model,LoraConfig,TaskType
from wm.cfg import WMCfg
from wm.bench.domain import DomainBenchmark

logging.basicConfig(level=logging.INFO,format="%(asctime)s %(name)s %(message)s")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--exa-key",default=None)
    ap.add_argument("--out",default="domain_bench_results.json")
    args=ap.parse_args()
    cfg=WMCfg()
    if args.exa_key:
        cfg.search.exa_api_key=args.exa_key
    cfg.train.base_model=args.model
    tok=AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(args.model)
    lora=LoraConfig(task_type=TaskType.CAUSAL_LM,r=16,lora_alpha=32,
                    lora_dropout=0.05,target_modules=["q_proj","v_proj"])
    model=get_peft_model(model,lora)
    bench=DomainBenchmark(cfg,model,tok)
    report=bench.run()
    print(bench.summary())
    with open(args.out,"w") as f:
        json.dump(report,f,indent=2,default=str)
    print(f"saved to {args.out}")

if __name__=="__main__":
    main()
