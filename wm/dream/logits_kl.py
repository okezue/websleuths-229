from __future__ import annotations
import torch
import torch.nn.functional as F

class LogitsKL:
    def __init__(self,temp:float=2.0):
        self._t=temp
    def loss(self,s_logits:torch.Tensor,t_logits:torch.Tensor)->torch.Tensor:
        s=F.log_softmax(s_logits/self._t,dim=-1)
        t=F.softmax(t_logits/self._t,dim=-1)
        kl=F.kl_div(s,t,reduction="none").sum(-1)
        return (kl.mean()*(self._t**2)).clamp_min(0.0)
