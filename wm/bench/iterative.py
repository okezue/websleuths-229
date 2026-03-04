from __future__ import annotations
import os,time,json,logging
from dataclasses import dataclass,field,asdict
from wm.bench.eval_harness import BenchScore,DomainEvalHarness
from wm.eval.anchor import AnchorEval
from wm.guard.drift import drift_kl
from wm.pipe.loop import AgenticPipeline
from wm.cfg import WMCfg

log=logging.getLogger(__name__)

TOPIC_SCHEDULE={
    "finance":["earnings ratio analysis SEC filings",
               "credit risk modeling Basel IV",
               "quantitative trading strategies",
               "ESG investing sustainability metrics",
               "derivatives pricing options"],
    "legal":["Fourth Amendment digital search privacy",
             "constitutional law Supreme Court precedent",
             "intellectual property AI copyright",
             "criminal procedure evidence rules",
             "international humanitarian law"],
    "chemistry":["organic reaction mechanisms catalysis",
                 "CRISPR gene editing molecular",
                 "computational chemistry molecular dynamics",
                 "green chemistry sustainable synthesis",
                 "polymer chemistry materials science"],
    "medicine":["clinical pharmacology drug interactions",
                "pathophysiology cardiovascular disease",
                "immunology vaccine mechanisms",
                "oncology targeted therapy",
                "neuroscience neurodegenerative disease"],
}

def flat_schedule(sched:dict[str,list[str]]|None=None)->list[tuple[str,str]]:
    sched=sched or TOPIC_SCHEDULE
    doms=list(sched.keys())
    mx=max(len(v) for v in sched.values())
    out=[]
    for i in range(mx):
        for d in doms:
            topics=sched[d]
            if i<len(topics):
                out.append((d,topics[i]))
    return out

def topic_domain(topic:str,sched:dict[str,list[str]]|None=None)->str:
    sched=sched or TOPIC_SCHEDULE
    for d,ts in sched.items():
        if topic in ts:return d
    return "unknown"

@dataclass
class TopicResult:
    idx:int
    topic:str
    domain:str
    pipe_result:dict=field(default_factory=dict)
    train_time:float=0.0
    domain_bench:BenchScore=field(default_factory=lambda:BenchScore("",0.0,0))
    retention:dict[str,BenchScore]=field(default_factory=dict)
    mmlu:dict[str,BenchScore]=field(default_factory=dict)
    anchor_nll:float=0.0
    anchor_delta:float=0.0
    drift_kl_val:float=0.0
    examples:list[dict]=field(default_factory=list)
    guard_accepted:bool=True

@dataclass
class CLReport:
    recipe:str=""
    baseline_bench:dict[str,BenchScore]=field(default_factory=dict)
    baseline_mmlu:dict[str,BenchScore]=field(default_factory=dict)
    baseline_anchor:float=0.0
    topics:list[TopicResult]=field(default_factory=list)
    final_bench:dict[str,BenchScore]=field(default_factory=dict)
    final_mmlu:dict[str,BenchScore]=field(default_factory=dict)
    final_anchor:float=0.0
    total_time:float=0.0

class IterativeCLBench:
    def __init__(self,cfg:WMCfg,model,tok,recipe_fn,recipe_name:str,
                 guard=None,ckpt_dir="/tmp/wm_checkpoints",
                 bench_n=100,mmlu_n=50):
        self._cfg=cfg
        self._m=model
        self._t=tok
        self._rfn=recipe_fn
        self._rn=recipe_name
        self._guard=guard
        self._ckpt=ckpt_dir
        self._bn=bench_n
        self._mn=mmlu_n
        self._base_snap=None
        self._domains=list(TOPIC_SCHEDULE.keys())
    def run(self,schedule=None)->CLReport:
        t0=time.time()
        dev=next(self._m.parameters()).device
        self._base_snap={k:v.clone() for k,v in self._m.state_dict().items()}
        harness=DomainEvalHarness(self._m,self._t,n_samples=self._bn,seed=42)
        log.info("baseline eval: domains + mmlu")
        bl_bench=harness.eval_all_domains(self._domains)
        bl_mmlu=harness.eval_all_mmlu(self._domains)
        anc_eval=AnchorEval(self._m,self._t)
        bl_anc=anc_eval.nll()
        report=CLReport(
            recipe=self._rn,
            baseline_bench=bl_bench,
            baseline_mmlu=bl_mmlu,
            baseline_anchor=bl_anc,
        )
        sched=flat_schedule(schedule)
        pipe=AgenticPipeline(self._cfg,self._m,self._t,self._rfn,self._guard)
        pipe.pace.state.tau_search=self._cfg.search_gate.tau_search
        pipe.pace.state.tau_ready=self._cfg.update_gate.tau_ready
        log.info("gate: tau_search=%.3f tau_ready=%.3f",pipe.pace.tau_search,pipe.pace.tau_ready)
        _tau_s=self._cfg.search_gate.tau_search
        _tau_r=self._cfg.update_gate.tau_ready
        for idx,(dom,topic) in enumerate(sched):
            pipe.pace.state.tau_search=_tau_s
            pipe.pace.state.tau_ready=_tau_r
            log.info(f"[{idx+1}/{len(sched)}] {dom}: {topic}")
            tt0=time.time()
            try:
                pr=pipe.run(topic)
                log.info(f"  pipe result: {pr}")
            except Exception as e:
                log.warning(f"pipe.run failed: {e}")
                pr={"action":"error","error":str(e)}
            tt=time.time()-tt0
            ga=pr.get("guard",True)
            if ga=="no_guard":ga=True
            slug=topic.replace(" ","_")[:30]
            ckp=os.path.join(self._ckpt,self._rn,f"{idx:03d}_{slug}")
            try:
                os.makedirs(ckp,exist_ok=True)
                self._m.save_pretrained(ckp)
            except Exception as e:
                log.warning(f"checkpoint save failed: {e}")
            db=harness.eval_domain(dom,n=self._bn)
            ret={}
            for d in self._domains:
                ret[d]=harness.eval_domain(d,n=self._bn)
            mmlu={}
            for d in self._domains:
                mmlu[d]=harness.eval_mmlu(d,n=self._mn)
            exs=harness.generate_examples(dom,n=3)
            cur_anc=anc_eval.nll()
            anc_d=cur_anc-bl_anc
            dk=0.0
            tr=TopicResult(
                idx=idx,topic=topic,domain=dom,
                pipe_result=pr,train_time=tt,
                domain_bench=db,retention=ret,mmlu=mmlu,
                anchor_nll=cur_anc,anchor_delta=anc_d,
                drift_kl_val=dk,examples=exs,
                guard_accepted=bool(ga),
            )
            report.topics.append(tr)
            log.info(f"  bench={db.acc:.4f} anc_d={anc_d:+.4f} dk={dk:.6f} guard={ga}")
        harness2=DomainEvalHarness(self._m,self._t,n_samples=self._bn,seed=42)
        report.final_bench=harness2.eval_all_domains(self._domains)
        report.final_mmlu=harness2.eval_all_mmlu(self._domains)
        report.final_anchor=anc_eval.nll()
        report.total_time=time.time()-t0
        try:pipe.close()
        except:pass
        return report
    def retention_matrix(self,report:CLReport)->dict[str,list[float]]:
        mx={}
        for d in self._domains:
            mx[d]=[]
        for tr in report.topics:
            for d in self._domains:
                sc=tr.retention.get(d)
                mx[d].append(sc.acc if sc else 0.0)
        return mx
    def summary(self,report:CLReport)->str:
        lines=[f"Recipe: {report.recipe}"]
        lines.append(f"Topics: {len(report.topics)}")
        lines.append(f"Total time: {report.total_time:.1f}s")
        lines.append(f"Baseline anchor: {report.baseline_anchor:.4f}")
        lines.append(f"Final anchor: {report.final_anchor:.4f}")
        lines.append(f"Anchor delta: {report.final_anchor-report.baseline_anchor:+.4f}")
        lines.append("")
        lines.append("Baseline benchmarks:")
        for d,bs in report.baseline_bench.items():
            lines.append(f"  {d}: {bs.acc:.4f} (n={bs.n})")
        lines.append("Final benchmarks:")
        for d,bs in report.final_bench.items():
            lines.append(f"  {d}: {bs.acc:.4f} (n={bs.n})")
        lines.append("")
        mx=self.retention_matrix(report)
        lines.append("Retention matrix (acc per step):")
        for d,vals in mx.items():
            vstr=" ".join(f"{v:.3f}" for v in vals)
            lines.append(f"  {d}: {vstr}")
        ga_n=sum(1 for t in report.topics if t.guard_accepted)
        lines.append(f"\nGuard accepted: {ga_n}/{len(report.topics)}")
        return "\n".join(lines)
    def save_report(self,report:CLReport,path:str):
        def _bs(b):
            if isinstance(b,BenchScore):return {"name":b.name,"acc":b.acc,"n":b.n}
            if isinstance(b,dict):return {k:_bs(v) for k,v in b.items()}
            return b
        d={
            "recipe":report.recipe,
            "baseline_bench":_bs(report.baseline_bench),
            "baseline_mmlu":_bs(report.baseline_mmlu),
            "baseline_anchor":report.baseline_anchor,
            "final_bench":_bs(report.final_bench),
            "final_mmlu":_bs(report.final_mmlu),
            "final_anchor":report.final_anchor,
            "total_time":report.total_time,
            "topics":[],
        }
        for t in report.topics:
            td={
                "idx":t.idx,"topic":t.topic,"domain":t.domain,
                "train_time":t.train_time,
                "domain_bench":_bs(t.domain_bench),
                "retention":_bs(t.retention),
                "mmlu":_bs(t.mmlu),
                "anchor_nll":t.anchor_nll,"anchor_delta":t.anchor_delta,
                "drift_kl":t.drift_kl_val,
                "examples":t.examples,
                "guard_accepted":t.guard_accepted,
                "pipe_result":{k:v for k,v in t.pipe_result.items()
                              if not isinstance(v,(bytes,))},
            }
            d["topics"].append(td)
        d["retention_matrix"]=self.retention_matrix(report)
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".",exist_ok=True)
        with open(path,"w") as f:
            json.dump(d,f,indent=2,default=str)
