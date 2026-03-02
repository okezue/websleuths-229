from __future__ import annotations
import random,torch

class DreamBuffer:
    def __init__(self,prompts:list[str],tok,max_len:int=128,n:int=4):
        self._p=prompts
        self._tok=tok
        self._ml=max_len
        self._n=n
    def sample(self,dev:torch.device)->dict[str,torch.Tensor]|None:
        if not self._p:return None
        batch=random.choices(self._p,k=self._n)
        enc=self._tok(batch,return_tensors="pt",truncation=True,
            max_length=self._ml,padding=True)
        return {k:v.to(dev) for k,v in enc.items()}
    def sample_texts(self,n:int=0)->list[str]:
        k=n or self._n
        if not self._p:return []
        return random.choices(self._p,k=k)
    @property
    def size(self)->int:
        return len(self._p)
