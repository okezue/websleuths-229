from __future__ import annotations
import logging
from datasets import Dataset
from wm.types import SearchResult,Chunk
from wm.cfg import WMCfg
from wm.search.agent import AgenticSearcher
from wm.graph.store import GraphStore
from wm.graph.dream_gen import gen_dream_prompts,gen_dream_bank
from wm.gate.search_gate import SearchGate
from wm.gate.update_gate import UpdateGate
from wm.pace.controller import PaceController
from wm.guard.orchestrator import UpdateGuard
from wm.adapt import adaptive_steps

log=logging.getLogger(__name__)

class AgenticPipeline:
    def __init__(self,cfg:WMCfg,model,tok,
                 recipe_fn=None,guard:UpdateGuard|None=None):
        self._cfg=cfg
        self._m=model
        self._t=tok
        self._recipe_fn=recipe_fn
        self._sg=SearchGate(cfg.search_gate)
        self._ug=UpdateGate(cfg.update_gate)
        self._pc=PaceController(cfg.pace)
        self._gs=GraphStore(cfg.graph.db_path)
        self._guard=guard
        self._dbank=None
    @property
    def graph_store(self)->GraphStore:
        return self._gs
    @property
    def dbank(self):
        return self._dbank
    @property
    def pace(self)->PaceController:
        return self._pc
    def run(self,topic:str)->dict:
        comms=self._gs.get_communities(only_param=True)
        sg_res=self._sg.check(topic,communities=comms,model=self._m,tok=self._t,
                              tau=self._pc.tau_search)
        if not sg_res.should_search:
            log.info("search gate: skip (%s)",sg_res.reason)
            return {"action":"skip_search","reason":sg_res.reason}
        searcher=AgenticSearcher(self._cfg.search,model=self._m,tok=self._t)
        sr=searcher.search(topic)
        self._gs.store_result(sr)
        self._pc.on_search()
        ug_res=self._ug.check(sr,model=self._m,tok=self._t,
                              tau_ready=self._pc.tau_ready)
        if not ug_res.should_update:
            log.info("update gate: skip (%s)",ug_res.reason)
            return {"action":"skip_update","reason":ug_res.reason,
                    "search":{"claims":len(sr.claims),"communities":len(sr.communities)}}
        if self._dbank is None:
            self._dbank=gen_dream_bank(sr.communities,sr.claims,self._t)
        else:
            self._dbank.add_episode(sr.communities,sr.claims)
        dreams=gen_dream_prompts(sr.communities,sr.claims)
        if sr.train_rows:
            ds_rows=sr.train_rows
            log.info("using %d Claude KG train_rows",len(ds_rows))
        else:
            ds_rows=[{"text":c.text,"authority":c.confidence} for c in sr.claims[:100]]
            if ds_rows:
                log.info("using %d claim-based train rows (no KG)",len(ds_rows))
            else:
                ds_rows=[{"text":c.text,"authority":c.authority} for c in sr.chunks]
                log.info("using %d raw chunk train rows (fallback)",len(ds_rows))
        if not ds_rows:
            log.warning("no training data available for topic")
            return {"action":"no_data","claims":len(sr.claims),"communities":len(sr.communities)}
        ds=Dataset.from_list(ds_rows)
        icl=self._cfg.iter_cl
        asteps=adaptive_steps(sr,icl.min_steps,icl.max_steps,
                              icl.step_alpha,icl.step_beta)
        result={"action":"train","claims":len(sr.claims),
                "communities":len(sr.communities),"dreams":len(dreams),
                "adaptive_steps":asteps,"train_rows":len(ds_rows),
                "train_source":"claude_kg" if sr.train_rows else "claims_or_chunks"}
        _db=self._dbank
        _rfn=lambda m,d,dp:self._recipe_fn(m,d,dp,steps=asteps,dbank=_db) if self._recipe_fn else None
        if self._guard and _rfn:
            gr=self._guard.guard(self._m,sr.chunks,
                                 _rfn,ds,dreams)
            result["guard"]=gr.accepted
            result["guard_reason"]=gr.reason
            if gr.accepted:
                for co in sr.communities:
                    self._gs.mark_parameterized(co.coid)
        elif _rfn:
            _rfn(self._m,ds,dreams)
            for co in sr.communities:
                self._gs.mark_parameterized(co.coid)
            result["guard"]="no_guard"
        self._pc.on_ft()
        return result
    def close(self):
        self._gs.close()
