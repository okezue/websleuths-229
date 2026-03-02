from __future__ import annotations
import torch
from wm.cfg import DreamCfg,MixCfg
from wm.dream.logits_kl import LogitsKL
from wm.dream.sampled import SampledKL

class PerStep:
    def should_dream(self,step:int,total:int)->bool:
        return True

class Interleave:
    def should_dream(self,step:int,total:int)->bool:
        return step%2==1

class TwoPhase:
    def __init__(self,ratio:float=0.5):
        self._r=ratio
    def should_dream(self,step:int,total:int)->bool:
        return step>=int(total*self._r)

_MODES={"logits_kl":LogitsKL,"sampled":SampledKL}
_SCHEDS={"per_step":lambda _:PerStep(),"interleave":lambda _:Interleave(),
    "two_phase":lambda c:TwoPhase(c.phase_ratio)}

class Dreamer:
    def __init__(self,dcfg:DreamCfg,mcfg:MixCfg):
        self._mode=_MODES[dcfg.mode](temp=dcfg.temp,**({} if dcfg.mode=="logits_kl" else {"n_samples":dcfg.n_samples}))
        self._sched=_SCHEDS[mcfg.sched](mcfg)
        self._w=dcfg.weight
    def step(self,step:int,total:int,s_logits:torch.Tensor,t_logits:torch.Tensor)->torch.Tensor|None:
        if not self._sched.should_dream(step,total):
            return None
        return self._w*self._mode.loss(s_logits,t_logits)
