from __future__ import annotations
import logging,json
from dataclasses import dataclass,field

log=logging.getLogger(__name__)

@dataclass
class CircuitResult:
    prompt:str=""
    n_features:int=0
    n_edges:int=0
    top_features:list[dict]=field(default_factory=list)
    scores:dict=field(default_factory=dict)
    graph_path:str=""

class CircuitAnalyzer:
    def __init__(self,model_name:str="google/gemma-2-2b",
                 transcoder_set:str="gemma",device:str|None=None):
        self._mn=model_name
        self._ts=transcoder_set
        self._dev=device
        self._model=None
    def _load(self):
        if self._model is not None:
            return
        try:
            from circuit_tracer import ReplacementModel
            import torch
            self._model=ReplacementModel.from_pretrained(
                model_name=self._mn,
                transcoder_set=self._ts,
                backend="transformerlens",
                dtype=torch.bfloat16,
                device=self._dev)
            log.info("circuit_tracer model loaded: %s",self._mn)
        except ImportError:
            log.warning("circuit_tracer not installed; pip install circuit-tracer")
            self._model=None
        except Exception as e:
            log.warning("circuit_tracer load failed: %s",e)
            self._model=None
    def attribute(self,prompt:str,max_features:int=50,
                  batch_size:int=256,save_path:str|None=None)->CircuitResult:
        self._load()
        if self._model is None:
            return CircuitResult(prompt=prompt)
        try:
            from circuit_tracer import attribute
            graph=attribute(
                prompt=prompt,model=self._model,
                max_n_logits=10,desired_logit_prob=0.95,
                batch_size=batch_size,max_feature_nodes=max_features,
                verbose=False)
            n_feat=len(graph.active_features) if hasattr(graph,"active_features") else 0
            adj=graph.adjacency_matrix if hasattr(graph,"adjacency_matrix") else None
            n_edges=int(adj.nonzero().shape[0]) if adj is not None else 0
            scores={}
            try:
                from circuit_tracer.graph import compute_graph_scores
                scores=compute_graph_scores(graph)
                if not isinstance(scores,dict):
                    scores={"score":float(scores)}
            except Exception:
                pass
            gp=""
            if save_path:
                graph.to_pt(save_path)
                gp=save_path
            return CircuitResult(prompt=prompt,n_features=n_feat,n_edges=n_edges,
                                  scores=scores,graph_path=gp)
        except Exception as e:
            log.warning("attribution failed: %s",e)
            return CircuitResult(prompt=prompt)
    def trace_domain_knowledge(self,prompts:list[str],
                                save_dir:str|None=None)->list[CircuitResult]:
        results=[]
        for i,p in enumerate(prompts):
            sp=f"{save_dir}/circuit_{i:03d}.pt" if save_dir else None
            results.append(self.attribute(p,save_path=sp))
        return results
    def compare_before_after(self,prompts:list[str],
                              before_results:list[CircuitResult]|None=None)->dict:
        after=self.trace_domain_knowledge(prompts)
        if before_results is None:
            return {"after":[{"prompt":r.prompt,"n_features":r.n_features,
                              "n_edges":r.n_edges,"scores":r.scores} for r in after]}
        diffs=[]
        for b,a in zip(before_results,after):
            diffs.append({
                "prompt":a.prompt,
                "features_delta":a.n_features-b.n_features,
                "edges_delta":a.n_edges-b.n_edges,
            })
        return {"diffs":diffs,"after":[{"prompt":r.prompt,"n_features":r.n_features,
                                         "n_edges":r.n_edges} for r in after]}
