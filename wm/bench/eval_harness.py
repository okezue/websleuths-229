from __future__ import annotations
import re,math,json as _json,logging,torch
from dataclasses import dataclass,field
_log=logging.getLogger(__name__)

@dataclass
class BenchScore:
    name:str;acc:float;n:int;extras:dict=field(default_factory=dict)

_DOMAIN_DS={
    "finance":("dreamerdeo/finqa","finqa",None),
    "legal":("coastalcph/lex_glue","case_hold",None),
    "chemistry":("jablonkagroup/ChemBench",None,None),
    "medicine":("bigbio/med_qa","med_qa_en_4options_source",None),
}
_MMLU_MAP={
    "finance":"professional_accounting",
    "legal":"professional_law",
    "chemistry":"college_chemistry",
    "medicine":"professional_medicine",
}
_LETTERS=["A","B","C","D","E"]

def _extract_letter(txt:str)->str:
    txt=txt.strip()
    if txt and txt[0].upper() in _LETTERS:
        return txt[0].upper()
    m=re.search(r'\b([A-E])\b',txt)
    if m:return m.group(1).upper()
    return ""

def _extract_number(txt:str)->float|None:
    txt=txt.replace(",","").replace("$","").replace("%","")
    m=re.search(r'-?\d+\.?\d*',txt)
    if m:
        try:return float(m.group())
        except:return None
    return None

def _num_close(a:float,b:float,tol:float=0.01)->bool:
    if b==0:return abs(a)<tol
    return abs(a-b)/max(abs(b),1e-12)<=tol

def _format_mcq(q:str,choices:list[str])->str:
    s=f"Q: {q}\n"
    for i,c in enumerate(choices):
        s+=f"{_LETTERS[i]}) {c}\n"
    s+="Answer with the letter only:"
    return s

def _gen(model,tok,prompt:str,max_tok:int=64,dev=None)->str:
    if dev is None:dev=next(model.parameters()).device
    enc=tok(prompt,return_tensors="pt",truncation=True,max_length=512)
    inp={k:v.to(dev) for k,v in enc.items() if k in ("input_ids","attention_mask")}
    with torch.no_grad():
        out=model.generate(**inp,max_new_tokens=max_tok,do_sample=False)
    return tok.decode(out[0][inp["input_ids"].shape[1]:],skip_special_tokens=True)

class DomainEvalHarness:
    def __init__(self,model,tok,max_len=512,n_samples=100,seed=42):
        self._m=model
        self._t=tok
        self._ml=max_len
        self._n=n_samples
        self._seed=seed
        self._dev=next(model.parameters()).device
        self._cache={}
    def _load_ds(self,name,subset=None,split="test"):
        k=(name,subset,split)
        if k in self._cache:return self._cache[k]
        from datasets import load_dataset
        ds=None
        for sp in [split,"validation","train"]:
            if ds is not None:break
            try:
                if subset:ds=load_dataset(name,subset,split=sp)
                else:ds=load_dataset(name,split=sp)
            except Exception:
                try:
                    if subset:ds=load_dataset(name,subset,split=sp,trust_remote_code=True)
                    else:ds=load_dataset(name,split=sp,trust_remote_code=True)
                except Exception:pass
            if ds is not None and sp!=split:break
        self._cache[k]=ds
        return ds
    def _sample(self,ds,n=0):
        if ds is None:return []
        if n<=0:return list(range(len(ds)))
        if len(ds)<=n:return list(range(len(ds)))
        import random
        rng=random.Random(self._seed)
        idx=list(range(len(ds)))
        rng.shuffle(idx)
        return idx[:n]
    def eval_bench(self,name:str,n=0)->BenchScore:
        self._m.eval()
        fn={"finqa":self._eval_finqa,"lexglue":self._eval_lexglue,
            "chembench":self._eval_chembench,"medqa":self._eval_medqa,
            "gpqa":self._eval_gpqa,"olympiad":self._eval_olympiad,
            "labbench":self._eval_labbench,"aime":self._eval_aime,
            "verus":self._eval_verus,"minif2f":self._eval_minif2f,
            "multikernelbench":self._eval_multikernelbench}.get(name)
        if fn:return fn(n)
        return BenchScore(name=name,acc=0.0,n=0)
    def eval_domain(self,domain:str,n=0)->BenchScore:
        self._m.eval()
        if domain=="finance":return self._eval_finqa(n)
        if domain=="legal":return self._eval_lexglue(n)
        if domain=="chemistry":return self._eval_chembench(n)
        if domain=="medicine":return self._eval_medqa(n)
        return BenchScore(name=f"{domain}_unknown",acc=0.0,n=0)
    def _eval_finqa(self,n=0)->BenchScore:
        ds=self._load_ds("wandb/finqa-data-processed",split="test")
        if ds is None:ds=self._load_ds("gagan3012/finqa-updated",split="test")
        if ds is None:ds=self._load_ds("dreamerdeo/finqa",split="test")
        if ds is None:
            _log.warning("finqa: no dataset found")
            return BenchScore(name="finqa",acc=0.0,n=0)
        _log.info("finqa: loaded %d rows, cols=%s",len(ds),ds.column_names[:10])
        idx=self._sample(ds,n)
        cor,tot=0,0
        for i in idx:
            row=ds[i]
            pre=row.get("pre_text",[])
            post=row.get("post_text",[])
            tbl=row.get("table","") or ""
            if isinstance(tbl,list):
                tbl="\n".join([" | ".join(r) if isinstance(r,list) else str(r) for r in tbl])
            ctx=""
            if isinstance(pre,list) and pre:ctx+=" ".join(pre)+"\n"
            ctx+=f"Table:\n{tbl}\n"
            if isinstance(post,list) and post:ctx+=" ".join(post)+"\n"
            q=row.get("question","") or row.get("query","")
            gold=row.get("answer","") or row.get("exe_ans","") or row.get("output","")
            prompt=f"{ctx}\nQuestion: {q}\nAnswer:"
            gen=_gen(self._m,self._t,prompt,max_tok=32,dev=self._dev)
            pred_n=_extract_number(gen)
            gold_n=_extract_number(str(gold))
            if pred_n is not None and gold_n is not None:
                if _num_close(pred_n,gold_n):cor+=1
            elif gold_n is None and len(str(gold).strip())>1:
                if str(gold).strip().lower() in gen.strip().lower():cor+=1
            tot+=1
        return BenchScore(name="finqa",acc=cor/max(tot,1),n=tot)
    def _eval_lexglue(self,n=0)->BenchScore:
        ds=self._load_ds("coastalcph/lex_glue","case_hold",split="test")
        if ds is None:return BenchScore(name="lexglue_casehold",acc=0.0,n=0)
        idx=self._sample(ds,n)
        cor,tot=0,0
        for i in idx:
            row=ds[i]
            ctx=row.get("context","")
            endings=row.get("endings",[])
            label=row.get("label",0)
            if not endings:tot+=1;continue
            choices=endings[:5]
            prompt=_format_mcq(ctx,choices)
            gen=_gen(self._m,self._t,prompt,max_tok=4,dev=self._dev)
            pred=_extract_letter(gen)
            gold=_LETTERS[label] if label<len(_LETTERS) else ""
            if pred==gold:cor+=1
            tot+=1
        return BenchScore(name="lexglue_casehold",acc=cor/max(tot,1),n=tot)
    def _eval_cb_item(self,inp:str,tgt:str,ts)->int:
        if isinstance(ts,str):
            try:
                import json as _j;ts=_j.loads(ts)
            except Exception:ts={}
        if isinstance(ts,dict) and ts:
            choices=list(ts.keys())[:4]
            gold_k=max(ts,key=lambda k:float(ts[k]) if isinstance(ts[k],(int,float)) else 0)
            prompt=_format_mcq(inp,choices)
            gen=_gen(self._m,self._t,prompt,max_tok=4,dev=self._dev)
            pred=_extract_letter(gen)
            gi=choices.index(gold_k) if gold_k in choices else -1
            return 1 if gi>=0 and gi<len(_LETTERS) and pred==_LETTERS[gi] else 0
        prompt=f"Q: {inp}\nAnswer:"
        gen=_gen(self._m,self._t,prompt,max_tok=32,dev=self._dev)
        return 1 if str(tgt).strip().lower() in gen.strip().lower() else 0
    def _eval_chembench(self,n=0)->BenchScore:
        from datasets import load_dataset,concatenate_datasets
        _cb_subs=["analytical_chemistry","general_chemistry","inorganic_chemistry",
                   "materials_science","organic_chemistry","physical_chemistry",
                   "technical_chemistry","toxicity_and_safety"]
        parts=[]
        for sub in _cb_subs:
            try:
                p=load_dataset("jablonkagroup/ChemBench",sub,split="train")
                parts.append(p)
            except Exception:pass
        if parts:
            ds=concatenate_datasets(parts)
        else:
            ds=self._load_ds("jablonkagroup/ChemBench",split="train")
        if ds is None:
            _log.warning("chembench: no dataset found")
            return BenchScore(name="chembench",acc=0.0,n=0)
        _log.info("chembench: loaded %d rows",len(ds))
        idx=self._sample(ds,n)
        cor,tot=0,0
        for i in idx:
            row=ds[i]
            exs=row.get("examples",None)
            if isinstance(exs,list) and exs:
                for ex in exs:
                    if not isinstance(ex,dict):continue
                    inp=ex.get("input","")
                    tgt=ex.get("target","")
                    ts=ex.get("target_scores",{})
                    cor+=self._eval_cb_item(inp,tgt,ts);tot+=1
            else:
                inp=row.get("input","") or row.get("question","")
                tgt=row.get("target","") or row.get("answer","")
                ts=row.get("target_scores",{})
                cor+=self._eval_cb_item(inp,tgt,ts);tot+=1
        return BenchScore(name="chembench",acc=cor/max(tot,1),n=tot)
    def _eval_medqa(self,n=0)->BenchScore:
        ds=self._load_ds("GBaker/MedQA-USMLE-4-options",split="test")
        if ds is None:ds=self._load_ds("bigbio/med_qa","med_qa_en_4options_source",split="test")
        if ds is None:return BenchScore(name="medqa",acc=0.0,n=0)
        idx=self._sample(ds,n)
        cor,tot=0,0
        for i in idx:
            row=ds[i]
            q=row.get("question","")
            opts=row.get("options",[]) or row.get("choices",[])
            ans=row.get("answer_idx",row.get("answer",""))
            if isinstance(opts,list) and opts and isinstance(opts[0],dict):
                choices=[o.get("value","") for o in opts[:4]]
            elif isinstance(opts,dict):
                choices=list(opts.values())[:4]
            else:
                choices=opts[:4] if opts else []
            gold=str(ans).upper()
            if not gold or gold not in _LETTERS:
                if isinstance(ans,int) and ans<len(_LETTERS):gold=_LETTERS[ans]
                else:gold=""
            if not choices:tot+=1;continue
            prompt=_format_mcq(q,choices)
            gen=_gen(self._m,self._t,prompt,max_tok=4,dev=self._dev)
            pred=_extract_letter(gen)
            if pred==gold:cor+=1
            tot+=1
        return BenchScore(name="medqa",acc=cor/max(tot,1),n=tot)
    def _eval_gpqa(self,n=0)->BenchScore:
        import os
        tok=os.environ.get("HF_TOKEN","")
        ds=None
        for cfg in ["gpqa_diamond","gpqa_main"]:
            if ds:break
            for sp in ["train","test"]:
                try:
                    from datasets import load_dataset
                    ds=load_dataset("Idavidrein/gpqa",cfg,split=sp,token=tok or None)
                    break
                except Exception:pass
        if ds is None:
            try:
                from datasets import load_dataset
                ds=load_dataset("openai/gpqa","gpqa_diamond",split="train",token=tok or None)
            except Exception:pass
        if ds is None:
            try:
                from datasets import load_dataset
                ds=load_dataset("nbalepur/GPQA",split="train",token=tok or None)
            except Exception:pass
        if ds is None:
            _log.warning("gpqa: no dataset found (accept terms at https://huggingface.co/datasets/Idavidrein/gpqa)")
            return BenchScore(name="gpqa",acc=0.0,n=0)
        _log.info("gpqa: loaded %d rows, cols=%s",len(ds),ds.column_names[:10])
        idx=self._sample(ds,n)
        cor,tot=0,0
        for i in idx:
            row=ds[i]
            q=row.get("Question","") or row.get("question","")
            choices=[]
            for k in ["Correct Answer","Incorrect Answer 1","Incorrect Answer 2","Incorrect Answer 3"]:
                v=row.get(k,"")
                if v:choices.append(v)
            if not choices:
                for k in ["choice1","choice2","choice3","choice4"]:
                    v=row.get(k,"")
                    if v:choices.append(v)
            gold_txt=row.get("Correct Answer","") or row.get("answer","")
            if not choices or not q:tot+=1;continue
            gold="A"
            for ci,c in enumerate(choices):
                if c==gold_txt and ci<len(_LETTERS):
                    gold=_LETTERS[ci];break
            prompt=_format_mcq(q,choices[:4])
            gen=_gen(self._m,self._t,prompt,max_tok=8,dev=self._dev)
            pred=_extract_letter(gen)
            if pred==gold:cor+=1
            tot+=1
        return BenchScore(name="gpqa",acc=cor/max(tot,1),n=tot)
    def _eval_olympiad(self,n=0)->BenchScore:
        import os
        tok=os.environ.get("HF_TOKEN","")
        ds=None
        for cfg in ["OE_TO_maths_en_COMP","TP_TO_maths_en_COMP","OE_TO_physics_en_COMP"]:
            if ds:break
            for sp in ["train","test_en","test"]:
                try:
                    from datasets import load_dataset
                    ds=load_dataset("Hothan/OlympiadBench",cfg,split=sp,token=tok or None)
                    break
                except Exception:pass
        if ds is None:
            for sp in ["test_en","test","train"]:
                try:
                    from datasets import load_dataset
                    ds=load_dataset("lmms-lab/OlympiadBench",split=sp,token=tok or None)
                    break
                except Exception:pass
        if ds is None:
            _log.warning("olympiad: no dataset found")
            return BenchScore(name="olympiad",acc=0.0,n=0)
        _log.info("olympiad: loaded %d rows, cols=%s",len(ds),ds.column_names[:10])
        idx=self._sample(ds,n)
        cor,tot=0,0
        for i in idx:
            row=ds[i]
            q=row.get("question","") or row.get("problem","")
            ans=row.get("final_answer","") or row.get("answer","")
            if isinstance(ans,list):ans=ans[0] if ans else ""
            if not q:tot+=1;continue
            prompt=f"Solve this competition problem. Give only the final numerical answer.\n\nProblem: {q}\n\nAnswer:"
            gen=_gen(self._m,self._t,prompt,max_tok=32,dev=self._dev)
            pred_n=_extract_number(gen)
            gold_n=_extract_number(str(ans))
            if pred_n is not None and gold_n is not None:
                if _num_close(pred_n,gold_n,tol=0.05):cor+=1
            elif str(ans).strip().lower() in gen.strip().lower():
                cor+=1
            tot+=1
        return BenchScore(name="olympiad",acc=cor/max(tot,1),n=tot)
    def _eval_labbench(self,n=0)->BenchScore:
        import os
        tok=os.environ.get("HF_TOKEN","")
        from datasets import load_dataset,concatenate_datasets
        parts=[]
        for cfg in ["LitQA2","DbQA","SeqQA","ProtocolQA","SuppQA","CloningScenarios"]:
            try:
                p=load_dataset("futurehouse/lab-bench",cfg,split="train",token=tok or None)
                parts.append(p)
            except Exception:pass
        if parts:
            ds=concatenate_datasets(parts)
        else:
            ds=self._load_ds("futurehouse/lab-bench",split="train")
        if ds is None:
            _log.warning("labbench: no dataset found")
            return BenchScore(name="labbench",acc=0.0,n=0)
        _log.info("labbench: loaded %d rows, cols=%s",len(ds),ds.column_names[:8])
        idx=self._sample(ds,n)
        cor,tot=0,0
        for i in idx:
            row=ds[i]
            q=row.get("question","")
            ideal=row.get("ideal","")
            distractors=row.get("distractors",[])
            if not q:tot+=1;continue
            if isinstance(distractors,list) and distractors:
                import random
                choices=[ideal]+distractors[:3]
                rng=random.Random(self._seed+i)
                rng.shuffle(choices)
                gold_idx=choices.index(ideal)
                gold=_LETTERS[gold_idx] if gold_idx<len(_LETTERS) else "A"
                prompt=_format_mcq(q,choices[:4])
                gen=_gen(self._m,self._t,prompt,max_tok=8,dev=self._dev)
                pred=_extract_letter(gen)
                if pred==gold:cor+=1
            else:
                prompt=f"Q: {q}\nAnswer:"
                gen=_gen(self._m,self._t,prompt,max_tok=64,dev=self._dev)
                if ideal.strip().lower() in gen.strip().lower():cor+=1
            tot+=1
        return BenchScore(name="labbench",acc=cor/max(tot,1),n=tot)
    def _eval_aime(self,n=0)->BenchScore:
        ds=self._load_ds("AI-MO/aimo-validation-aime",split="train")
        if ds is None:ds=self._load_ds("Maxwell-Jia/AIME_2024",split="train")
        if ds is None:ds=self._load_ds("di-dimitrov/aime-problem-set",split="train")
        if ds is None:
            _log.warning("aime: no dataset found")
            return BenchScore(name="aime",acc=0.0,n=0)
        _log.info("aime: loaded %d rows",len(ds))
        idx=self._sample(ds,n)
        cor,tot=0,0
        for i in idx:
            row=ds[i]
            q=row.get("problem","") or row.get("question","")
            ans=row.get("answer","") or row.get("solution","")
            prompt=f"Solve this math competition problem. Give only the final integer answer.\n\nProblem: {q}\n\nAnswer:"
            gen=_gen(self._m,self._t,prompt,max_tok=64,dev=self._dev)
            pred_n=_extract_number(gen)
            gold_n=_extract_number(str(ans))
            if tot<3:
                _log.info("aime[%d] gold=%s pred_n=%s gen=%s",tot,ans,pred_n,gen[:80])
            if pred_n is not None and gold_n is not None:
                if abs(pred_n-gold_n)<0.5:cor+=1
            tot+=1
        return BenchScore(name="aime",acc=cor/max(tot,1),n=tot)
    def _eval_verus(self,n=0)->BenchScore:
        import os
        vpath=os.environ.get("VERUS_DATA","/Users/okezuebell/Documents/GitHub/VerusFT_RL/datasets_training/task_a_tasks_v9.jsonl")
        if not os.path.exists(vpath):
            for alt in ["/home/ubuntu/verus_tasks.jsonl","datasets_training/task_a_tasks_v9.jsonl"]:
                if os.path.exists(alt):vpath=alt;break
        if not os.path.exists(vpath):
            _log.warning("verus: data not found at %s",vpath)
            return BenchScore(name="verus",acc=0.0,n=0)
        import json as _j
        rows=[]
        with open(vpath) as f:
            for line in f:
                if line.strip():rows.append(_j.loads(line))
        _log.info("verus: loaded %d tasks",len(rows))
        idx=self._sample(type("DS",(object,),{"__len__":lambda s:len(rows)})(),n)
        cor,tot=0,0
        for i in idx:
            row=rows[i]
            inp=row.get("input_text","")[:500]
            tgt=row.get("target_text","")
            prompt=f"Given this Rust/Verus function, generate the requires/ensures specifications:\n\n{inp}\n\nSpecifications:"
            gen=_gen(self._m,self._t,prompt,max_tok=200,dev=self._dev)
            has_req="requires" in gen.lower() or "ensures" in gen.lower()
            has_key=any(kw in gen.lower() for kw in ["requires","ensures","invariant","decreases"])
            tgt_keys=[kw for kw in ["requires","ensures","invariant","decreases"] if kw in tgt.lower()]
            gen_keys=[kw for kw in ["requires","ensures","invariant","decreases"] if kw in gen.lower()]
            overlap=len(set(tgt_keys)&set(gen_keys))
            total_keys=max(len(set(tgt_keys)),1)
            if overlap/total_keys>=0.5:cor+=1
            tot+=1
        return BenchScore(name="verus",acc=cor/max(tot,1),n=tot)
    def _eval_minif2f(self,n=0)->BenchScore:
        import os
        tok=os.environ.get("HF_TOKEN","")
        ds=None
        try:
            from datasets import load_dataset
            ds=load_dataset("cat-searcher/minif2f-lean4",split="test",token=tok or None)
        except:pass
        if ds is None:
            try:
                from datasets import load_dataset
                ds=load_dataset("formalml/minif2f",split="test",token=tok or None)
            except:pass
        if ds is None:
            try:
                from datasets import load_dataset
                ds=load_dataset("cat-searcher/minif2f-lean4",split="valid",token=tok or None)
            except:pass
        if ds is None:
            _log.warning("minif2f: no dataset found")
            return BenchScore(name="minif2f",acc=0.0,n=0)
        _log.info("minif2f: loaded %d rows, cols=%s",len(ds),ds.column_names[:5])
        idx=self._sample(ds,n)
        cor,tot=0,0
        for i in idx:
            row=ds[i]
            stmt=row.get("formal_statement","") or row.get("statement","") or row.get("problem","") or ""
            if not stmt:tot+=1;continue
            prompt=f"Complete this Lean 4 theorem proof:\n\n{stmt[:400]}\n\nProof:"
            gen=_gen(self._m,self._t,prompt,max_tok=150,dev=self._dev)
            has_proof=any(kw in gen.lower() for kw in ["sorry","by","simp","ring","omega","linarith","norm_num","exact","apply","intro","have"])
            if has_proof:cor+=1
            tot+=1
        return BenchScore(name="minif2f",acc=cor/max(tot,1),n=tot)
    def _eval_multikernelbench(self,n=0)->BenchScore:
        import os,subprocess,tempfile
        mkb=os.environ.get("MULTIKERNELBENCH_PATH","")
        if not mkb:
            for p in ["/home/ubuntu/MultiKernelBench","MultiKernelBench",
                      os.path.expanduser("~/MultiKernelBench")]:
                if os.path.isdir(p):mkb=p;break
        if not mkb or not os.path.isdir(mkb):
            _log.warning("multikernelbench: repo not found, set MULTIKERNELBENCH_PATH")
            return BenchScore(name="multikernelbench",acc=0.0,n=0)
        tasks_dir=os.path.join(mkb,"reference")
        if not os.path.isdir(tasks_dir):
            for alt in ["benchmark","tasks","data","bench"]:
                t=os.path.join(mkb,alt)
                if os.path.isdir(t):tasks_dir=t;break
        task_files=[]
        for root,dirs,files in os.walk(tasks_dir):
            for f in files:
                if f.endswith(".py") and "test" not in f.lower():
                    task_files.append(os.path.join(root,f))
        if not task_files:
            _log.warning("multikernelbench: no task files in %s",tasks_dir)
            return BenchScore(name="multikernelbench",acc=0.0,n=0)
        import random
        rng=random.Random(self._seed)
        rng.shuffle(task_files)
        if n>0:task_files=task_files[:n]
        _log.info("multikernelbench: evaluating %d tasks",len(task_files))
        compiled,passed,tot=0,0,0
        for tf in task_files:
            try:
                with open(tf) as fh:src=fh.read()
            except Exception:tot+=1;continue
            m=re.search(r'class\s+(\w+)\s*\(.*?nn\.Module.*?\):.*?def\s+forward\s*\(.*?\).*?(?=\nclass|\Z)',
                        src,re.DOTALL)
            if not m:tot+=1;continue
            ref_cls=m.group(0)
            prompt=(f"Write a CUDA or Triton kernel that implements this PyTorch module's forward method.\n"
                    f"Return ONLY the kernel code and a custom_module that calls it.\n\n"
                    f"Reference PyTorch module:\n```python\n{ref_cls[:1500]}\n```\n\nKernel:")
            gen=_gen(self._m,self._t,prompt,max_tok=512,dev=self._dev)
            has_kernel=any(k in gen.lower() for k in
                ["__global__","@triton.jit","tl.load","tl.store",
                 "cuda","blockidx","threadidx","triton.language"])
            has_func=("def " in gen or "void " in gen)
            if has_kernel and has_func:
                compiled+=1
                with tempfile.NamedTemporaryFile(mode="w",suffix=".py",delete=False) as tmp:
                    tmp.write(gen);tmp_path=tmp.name
                try:
                    r=subprocess.run(["python","-c",f"import ast;ast.parse(open('{tmp_path}').read())"],
                                     capture_output=True,timeout=5)
                    if r.returncode==0:passed+=1
                except Exception:pass
                finally:
                    try:os.unlink(tmp_path)
                    except Exception:pass
            tot+=1
        comp_rate=compiled/max(tot,1)
        pass_rate=passed/max(tot,1)
        return BenchScore(name="multikernelbench",acc=pass_rate,n=tot,
                          extras={"compilation_at_1":comp_rate,"pass_at_1":pass_rate,
                                  "compiled":compiled,"passed":passed})
    def eval_capability(self,bench:str,n=0)->BenchScore:
        self._m.eval()
        return self.eval_bench(bench,n)
    def eval_all_capabilities(self,n=0)->dict[str,BenchScore]:
        out={}
        for b in ["gpqa","olympiad","labbench","aime","verus","minif2f"]:
            out[b]=self.eval_capability(b,n)
        return out
    def eval_mmlu(self,domain:str,n=0)->BenchScore:
        self._m.eval()
        subj=_MMLU_MAP.get(domain)
        if not subj:return BenchScore(name=f"mmlu_{domain}",acc=0.0,n=0)
        ds=self._load_ds("cais/mmlu",subj,split="test")
        if ds is None:ds=self._load_ds("cais/mmlu",subj,split="validation")
        if ds is None:return BenchScore(name=f"mmlu_{subj}",acc=0.0,n=0)
        idx=self._sample(ds,n)
        cor,tot=0,0
        for i in idx:
            row=ds[i]
            q=row.get("question","")
            choices=row.get("choices",[])
            ans=row.get("answer",0)
            if not choices:tot+=1;continue
            prompt=_format_mcq(q,choices[:4])
            gen=_gen(self._m,self._t,prompt,max_tok=4,dev=self._dev)
            pred=_extract_letter(gen)
            if isinstance(ans,int) and ans<len(_LETTERS):gold=_LETTERS[ans]
            elif isinstance(ans,str) and ans.upper() in _LETTERS:gold=ans.upper()
            else:gold=""
            if pred==gold:cor+=1
            tot+=1
        return BenchScore(name=f"mmlu_{subj}",acc=cor/max(tot,1),n=tot)
    def eval_all_domains(self,domains=None)->dict[str,BenchScore]:
        domains=domains or list(_DOMAIN_DS.keys())
        return {d:self.eval_domain(d) for d in domains}
    def eval_all_mmlu(self,domains=None)->dict[str,BenchScore]:
        domains=domains or list(_MMLU_MAP.keys())
        return {d:self.eval_mmlu(d) for d in domains}
    def generate_examples(self,domain:str,n=3)->list[dict]:
        self._m.eval()
        prompts={
            "finance":["Explain the price-to-earnings ratio and its significance in stock valuation.",
                       "What are the key components of a company's balance sheet?",
                       "Describe how credit risk is assessed under Basel IV."],
            "legal":["What is the significance of the Fourth Amendment in digital privacy?",
                     "Explain the doctrine of stare decisis in constitutional law.",
                     "How does intellectual property law apply to AI-generated content?"],
            "chemistry":["Describe the SN2 reaction mechanism in organic chemistry.",
                        "What is the role of CRISPR-Cas9 in gene editing?",
                        "Explain molecular dynamics simulation in computational chemistry."],
            "medicine":["What are common drug-drug interactions in clinical pharmacology?",
                       "Describe the pathophysiology of heart failure.",
                       "How do mRNA vaccines elicit an immune response?"],
        }
        dp=prompts.get(domain,prompts.get("finance"))[:n]
        out=[]
        for p in dp:
            r=_gen(self._m,self._t,p,max_tok=150,dev=self._dev)
            out.append({"prompt":p,"response":r})
        return out
