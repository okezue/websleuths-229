#!/usr/bin/env python3
import os,sys,argparse,torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel

def main():
    ap=argparse.ArgumentParser(description="Interactive chat with checkpointed model")
    ap.add_argument("--model",default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--adapter",default=None,help="path to LoRA adapter checkpoint")
    ap.add_argument("--max-tokens",type=int,default=256)
    ap.add_argument("--temp",type=float,default=0.7)
    ap.add_argument("--hf-token",default=os.environ.get("HF_TOKEN",""))
    args=ap.parse_args()
    if args.hf_token:os.environ["HF_TOKEN"]=args.hf_token

    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model} on {dev}...")
    tok=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True)
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    base=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,
                                               trust_remote_code=True).to(dev)
    if args.adapter:
        print(f"Loading adapter from {args.adapter}...")
        base=PeftModel.from_pretrained(base,args.adapter).to(dev)
    base.eval()
    print(f"Ready. Type 'quit' to exit, 'compare' to see base vs adapted side-by-side.\n")

    compare_mode=False
    base_model=None
    while True:
        try:
            prompt=input(">>> ").strip()
        except (EOFError,KeyboardInterrupt):
            print();break
        if not prompt:continue
        if prompt.lower()=="quit":break
        if prompt.lower()=="compare":
            if args.adapter and base_model is None:
                print("Loading base model for comparison...")
                base_model=AutoModelForCausalLM.from_pretrained(
                    args.model,torch_dtype=torch.bfloat16,
                    trust_remote_code=True).to(dev)
                base_model.eval()
                print("Compare mode ON. Next prompt will show both outputs.\n")
            elif base_model is not None:
                del base_model;base_model=None
                torch.cuda.empty_cache()
                print("Compare mode OFF.\n")
            else:
                print("No adapter loaded, nothing to compare.\n")
            continue

        enc=tok(prompt,return_tensors="pt",truncation=True,max_length=512)
        inp={k:v.to(dev) for k,v in enc.items() if k in ("input_ids","attention_mask")}

        with torch.no_grad():
            out=base.generate(**inp,max_new_tokens=args.max_tokens,
                              do_sample=args.temp>0,temperature=args.temp if args.temp>0 else None,
                              top_p=0.95 if args.temp>0 else None)
        resp=tok.decode(out[0][inp["input_ids"].shape[1]:],skip_special_tokens=True)

        if base_model is not None:
            with torch.no_grad():
                out_b=base_model.generate(**inp,max_new_tokens=args.max_tokens,
                                          do_sample=args.temp>0,temperature=args.temp if args.temp>0 else None,
                                          top_p=0.95 if args.temp>0 else None)
            resp_b=tok.decode(out_b[0][inp["input_ids"].shape[1]:],skip_special_tokens=True)
            print(f"\n[BASE]    {resp_b.strip()}")
            print(f"\n[ADAPTED] {resp.strip()}\n")
        else:
            print(f"\n{resp.strip()}\n")

if __name__=="__main__":
    main()
