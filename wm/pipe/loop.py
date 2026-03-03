from __future__ import annotations
import logging
from datasets import Dataset
from wm.types import SearchResult,Chunk
from wm.cfg import WMCfg
from wm.search.agent import AgenticSearcher
from wm.graph.store import GraphStore
from wm.graph.dream_gen import gen_dream_prompts
from wm.gate.search_gate import SearchGate
from wm.gate.update_gate import UpdateGate
from wm.pace.controller import PaceController
from wm.guard.orchestrator import UpdateGuard

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
    @property
    def graph_store(self)->GraphStore:
        return self._gs
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
        dreams=gen_dream_prompts(sr.communities,sr.claims)
        ds_rows=[{"text":c.text,"authority":c.authority} for c in sr.chunks]
        if not ds_rows:
            ds_rows=[{"text":c.text,"authority":1.0} for c in sr.claims[:50]]
        ds=Dataset.from_list(ds_rows)
        result={"action":"train","claims":len(sr.claims),
                "communities":len(sr.communities),"dreams":len(dreams)}
        if self._guard and self._recipe_fn:
            gr=self._guard.guard(self._m,sr.chunks,
                                 self._recipe_fn,ds,dreams)
            result["guard"]=gr.accepted
            result["guard_reason"]=gr.reason
            if gr.accepted:
                for co in sr.communities:
                    self._gs.mark_parameterized(co.coid)
        elif self._recipe_fn:
            self._recipe_fn(self._m,ds,dreams)
            for co in sr.communities:
                self._gs.mark_parameterized(co.coid)
            result["guard"]="no_guard"
        self._pc.on_ft()
        return result
    def close(self):
        self._gs.close()
