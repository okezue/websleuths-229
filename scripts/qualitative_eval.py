#!/usr/bin/env python3
import os,sys,json,time,torch,logging
logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s",stream=sys.stdout)
log=logging.getLogger("qual_eval")

OPEN_ENDED_PROBES={
    "finance":[
        "Explain how a leveraged buyout works and what risks are involved.",
        "A company has $200M revenue growing 15% YoY with 20% margins. Should an investor buy at 25x P/E? Why or why not?",
        "What caused the 2008 financial crisis and what regulatory changes followed?",
        "How do you hedge currency risk in an international bond portfolio?",
        "Walk me through how to read a 10-K filing and what red flags to look for.",
    ],
    "legal":[
        "Explain the difference between strict liability and negligence in tort law.",
        "A tech company scrapes publicly available data. Under what circumstances could this violate copyright law?",
        "How does qualified immunity work and why is it controversial?",
        "What are the key differences between civil and criminal proceedings?",
        "Explain how the commerce clause has been interpreted to expand federal power.",
    ],
    "chemistry":[
        "Explain why water has such a high boiling point compared to H2S.",
        "Design a synthesis route for aspirin from benzene. What reagents would you need?",
        "Why do transition metals form colored compounds?",
        "Explain how a buffer solution works and calculate the pH of an acetate buffer.",
        "What's the difference between SN1 and SN2 mechanisms and when does each occur?",
    ],
    "medicine":[
        "A 45-year-old diabetic patient presents with numbness in their feet. What's your differential diagnosis?",
        "Explain the mechanism of action of SSRIs and why they take weeks to work.",
        "How does the immune system distinguish self from non-self?",
        "What are the physiological effects of chronic alcohol abuse on the liver?",
        "Explain why antibiotic resistance is developing and what can be done about it.",
    ],
    "reasoning":[
        "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
        "A bat and ball cost $1.10 together. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "Three friends split a $30 hotel room. The clerk gives $5 back. The bellboy keeps $2 and returns $1 each. Each paid $9, totaling $27. The bellboy has $2. That's $29. Where's the missing dollar?",
    ],
    "formal_verification":[
        "Write preconditions and postconditions for a binary search function in Rust that takes a sorted array and target value.",
        "What loop invariant would you need to verify that a summation loop correctly computes the sum of an array?",
        "Explain how to prove in Lean 4 that reversing a list twice gives back the original list.",
        "Given a function that removes duplicates from a sorted array in-place, what specifications would you write?",
        "What decreases clause is needed to prove termination of a recursive GCD function?",
    ],
}

def gen(model,tok,prompt,max_tok=300,dev=None):
    if dev is None:dev=next(model.parameters()).device
    model.eval()
    enc=tok(prompt,return_tensors="pt",truncation=True,max_length=512)
    inp={k:v.to(dev) for k,v in enc.items() if k in ("input_ids","attention_mask")}
    with torch.no_grad():
        out=model.generate(**inp,max_new_tokens=max_tok,do_sample=False)
    return tok.decode(out[0][inp["input_ids"].shape[1]:],skip_special_tokens=True)

def judge_with_gpt(prompt,response,openai_key,model="gpt-5.4"):
    from openai import OpenAI
    c=OpenAI(api_key=openai_key,timeout=120,max_retries=1)
    judge_prompt=f"""Rate this language model response on a scale of 1-10 for each criterion.

Question: {prompt}

Model's response: {response[:1000]}

Rate:
1. **Accuracy** (1-10): Are the facts correct?
2. **Depth** (1-10): Does it show real understanding vs surface-level?
3. **Specificity** (1-10): Does it use domain-specific terms and concrete examples?
4. **Reasoning** (1-10): Does it show logical step-by-step thinking?
5. **Completeness** (1-10): Does it fully address the question?

Return JSON only:
{{"accuracy": 1-10, "depth": 1-10, "specificity": 1-10, "reasoning": 1-10, "completeness": 1-10, "overall": 1-10, "brief_assessment": "1-2 sentences"}}"""
    try:
        r=c.responses.create(model=model,input=judge_prompt,
                              reasoning={"effort":"high"})
        txt=r.output_text.strip()
        if txt.startswith("```"):
            lines=txt.split("\n")
            txt="\n".join(l for l in lines if not l.strip().startswith("```"))
        import re
        txt=re.sub(r',\s*}','}',txt)
        s=txt.find("{");e=txt.rfind("}")+1
        if s>=0 and e>s:
            return json.loads(txt[s:e])
    except Exception as ex:
        log.warning("judge failed: %s",ex)
    return {}

def run_eval(model,tok,openai_key,tag="",od="/tmp/qual_eval",dev=None):
    os.makedirs(od,exist_ok=True)
    results={"tag":tag,"timestamp":time.strftime("%Y-%m-%d %H:%M:%S"),"probes":{}}
    total_score=0;n_scored=0
    for domain,probes in OPEN_ENDED_PROBES.items():
        results["probes"][domain]=[]
        for p in probes:
            log.info("[%s] %s: %s",tag,domain,p[:50])
            resp=gen(model,tok,p,dev=dev)
            log.info("  response: %s",resp[:120])
            judgment=judge_with_gpt(p,resp,openai_key) if openai_key else {}
            if judgment.get("overall"):
                total_score+=judgment["overall"]
                n_scored+=1
                log.info("  GPT judge: overall=%d/10 — %s",
                         judgment["overall"],judgment.get("brief_assessment",""))
            results["probes"][domain].append({
                "prompt":p,"response":resp,"judgment":judgment})
    results["avg_score"]=total_score/max(n_scored,1)
    results["n_scored"]=n_scored
    with open(f"{od}/{tag or 'eval'}.json","w") as f:
        json.dump(results,f,indent=2)
    log.info("[%s] avg GPT judge score: %.1f/10 (%d probes scored)",
             tag,results["avg_score"],n_scored)
    return results

def compare(before,after):
    diffs=[]
    for domain in OPEN_ENDED_PROBES:
        bp=before.get("probes",{}).get(domain,[])
        ap=after.get("probes",{}).get(domain,[])
        for i in range(min(len(bp),len(ap))):
            bj=bp[i].get("judgment",{})
            aj=ap[i].get("judgment",{})
            if bj.get("overall") and aj.get("overall"):
                d=aj["overall"]-bj["overall"]
                diffs.append({"domain":domain,"prompt":bp[i]["prompt"][:50],
                              "before":bj["overall"],"after":aj["overall"],"delta":d,
                              "before_assessment":bj.get("brief_assessment",""),
                              "after_assessment":aj.get("brief_assessment","")})
    diffs.sort(key=lambda x:x["delta"],reverse=True)
    return diffs

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True)
    ap.add_argument("--checkpoint",default=None)
    ap.add_argument("--tag",default="eval")
    ap.add_argument("--out",default="/tmp/qual_eval")
    ap.add_argument("--openai-key",default=os.environ.get("OPENAI_API_KEY",""))
    ap.add_argument("--hf-token",default=os.environ.get("HF_TOKEN",""))
    args=ap.parse_args()
    if args.hf_token:os.environ["HF_TOKEN"]=args.hf_token

    from transformers import AutoModelForCausalLM,AutoTokenizer
    tok=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True)
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    m=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,
        trust_remote_code=True).to("cuda:0")
    if args.checkpoint:
        from peft import PeftModel
        m=PeftModel.from_pretrained(m,args.checkpoint)
        log.info("loaded checkpoint: %s",args.checkpoint)

    run_eval(m,tok,args.openai_key,tag=args.tag,od=args.out)
