from __future__ import annotations
import torch
import torch.nn.functional as F

class SampledKL:
    def __init__(self,temp:float=2.0,n_samples:int=16):
        self._t=temp
        self._n=n_samples
    def loss(self,s_logits:torch.Tensor,t_logits:torch.Tensor)->torch.Tensor:
        t_probs=F.softmax(t_logits/self._t,dim=-1)
        samples=torch.multinomial(t_probs.view(-1,t_probs.shape[-1]),self._n,replacement=True)
        s_lp=F.log_softmax(s_logits/self._t,dim=-1)
        t_lp=F.log_softmax(t_logits/self._t,dim=-1)
        s_flat=s_lp.view(-1,s_lp.shape[-1])
        t_flat=t_lp.view(-1,t_lp.shape[-1])
        s_sel=torch.gather(s_flat,1,samples)
        t_sel=torch.gather(t_flat,1,samples)
        return ((t_sel-s_sel).mean())*(self._t**2)
