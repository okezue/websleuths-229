from __future__ import annotations
import copy
from dataclasses import dataclass,field
from wm.types import Chunk
from wm.eval.probes import ProbeBuilder
from wm.eval.anchor import AnchorEval
from wm.eval.core import Evaluator

@dataclass
class BenchResult:
    recipe:str
    delta_qa:float=0.0
    delta_anchor:float=0.0
    drift_kl:float=0.0
    pre_qa:float=0.0
    post_qa:float=0.0

class RecipeBenchmark:
    def __init__(self,tok,anchors:list[str]|None=None,max_len:int=128):
        self._tok=tok
        self._anchors=anchors
        self._ml=max_len
    def run(self,ep_ds,chunks:list[Chunk],dream_prompts:list[str],
            recipes:dict)->list[BenchResult]:
        pb=ProbeBuilder()
        probes=pb.build_all(chunks)
        results=[]
        for name,fn in recipes.items():
            m_copy=copy.deepcopy(ep_ds["model"])
            ev=Evaluator(m_copy,self._tok,max_len=self._ml)
            pre=[ev._probe_score(p) for p in probes] if probes else []
            anc=AnchorEval(m_copy,self._tok,self._anchors)
            pre_nll=anc.nll()
            fn(m_copy,ep_ds["dataset"],dream_prompts)
            post=[ev._probe_score(p) for p in probes] if probes else []
            post_nll=anc.nll()
            pre_mean=sum(pre)/max(len(pre),1) if pre else 0.0
            post_mean=sum(post)/max(len(post),1) if post else 0.0
            results.append(BenchResult(
                recipe=name,
                delta_qa=post_mean-pre_mean,
                delta_anchor=post_nll-pre_nll,
                pre_qa=pre_mean,post_qa=post_mean,
            ))
        return results
    def report(self,results:list[BenchResult])->str:
        lines=["Recipe          | delta_qa | delta_anchor | drift_kl",
               "----------------|----------|--------------|--------"]
        for r in results:
            lines.append(f"{r.recipe:<16}| {r.delta_qa:+.4f}  | {r.delta_anchor:+.4f}      | {r.drift_kl:.4f}")
        return "\n".join(lines)
