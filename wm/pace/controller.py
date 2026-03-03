from __future__ import annotations
from wm.types import PaceState
from wm.cfg import PaceCfg

class PaceController:
    def __init__(self,cfg:PaceCfg):
        self._cfg=cfg
        self._st=PaceState()
    @property
    def state(self)->PaceState:
        return self._st
    @property
    def tau_search(self)->float:
        return self._st.tau_search
    @property
    def tau_ready(self)->float:
        return self._st.tau_ready
    def on_search(self):
        self._st.web_calls+=1
        self._st.lam_web=max(0.0,
            self._st.lam_web+self._cfg.eta*(
                self._st.web_calls/max(self._cfg.web_budget,1)-1))
        self._st.tau_search=0.5+0.3*min(1.0,self._st.lam_web)
    def on_ft(self):
        self._st.ft_calls+=1
        self._st.lam_ft=max(0.0,
            self._st.lam_ft+self._cfg.eta*(
                self._st.ft_calls/max(self._cfg.ft_budget,1)-1))
        self._st.tau_ready=0.6+0.3*min(1.0,self._st.lam_ft)
    def reset(self):
        self._st=PaceState()
