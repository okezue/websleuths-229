from __future__ import annotations
import copy,logging
from wm.types import Chunk
from wm.cfg import WMCfg,DomainBenchCfg
from wm.eval.core import Evaluator
from wm.eval.anchor import AnchorEval
from wm.eval.probes import ProbeBuilder
from wm.pipe.loop import AgenticPipeline

log=logging.getLogger(__name__)

class DomainBenchmark:
    def __init__(self,cfg:WMCfg,model,tok,recipe_fn=None):
        self._cfg=cfg
        self._base=copy.deepcopy(model)
        self._m=model
        self._t=tok
        self._rfn=recipe_fn
        self._dcfg=cfg.domain_bench
        self._results:dict[str,dict]={}
    def _eval_probes(self,chunks:list[Chunk])->dict[str,float]:
        if not chunks:return {"hit_rate":0.0,"mean_f1":0.0,"n_probes":0}
        pb=ProbeBuilder()
        probes=pb.build_all(chunks)
        ev=Evaluator(self._m,self._t,max_len=128)
        return ev.evaluate_probes(probes)
    def _anchor_nll(self)->float:
        anc=AnchorEval(self._m,self._t)
        return anc.nll()
    def run(self)->dict:
        report={}
        base_nll=self._anchor_nll()
        report["baseline"]={"anchor_nll":base_nll}
        domain_chunks:dict[str,list[Chunk]]={}
        pipe=AgenticPipeline(self._cfg,self._m,self._t,recipe_fn=self._rfn)
        for dom in self._dcfg.domains:
            topic=self._dcfg.topics.get(dom,dom)
            log.info("domain benchmark: %s",dom)
            res=pipe.run(topic)
            sr_chunks=[]
            for c in pipe.graph_store.get_claims():
                sr_chunks.append(Chunk(eid=c.eid,idx=c.chunk_idx,text=c.text))
            domain_chunks[dom]=sr_chunks[-self._dcfg.probes_per_domain*3:]
            dom_report={"pipeline":res}
            dom_report["probes"]=self._eval_probes(domain_chunks[dom])
            for prev_dom,prev_chunks in domain_chunks.items():
                if prev_dom==dom:continue
                dom_report[f"retention_{prev_dom}"]=self._eval_probes(prev_chunks)
            dom_report["anchor_nll"]=self._anchor_nll()
            dom_report["anchor_delta"]=dom_report["anchor_nll"]-base_nll
            report[dom]=dom_report
        pipe.close()
        self._results=report
        return report
    def summary(self)->str:
        lines=["Domain Benchmark Summary","="*40]
        for k,v in self._results.items():
            lines.append(f"\n{k}:")
            if isinstance(v,dict):
                for sk,sv in v.items():
                    lines.append(f"  {sk}: {sv}")
        return "\n".join(lines)
