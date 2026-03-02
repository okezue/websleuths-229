from __future__ import annotations
from wm.types import TrainResult

class TinkerBack:
    def __init__(self,cfg=None):
        self._cfg=cfg
        self._available=False
    @property
    def available(self)->bool:
        return self._available
    def setup(self,model_name:str,lora_cfg:object,out_dir:str):
        raise RuntimeError("Tinker backend not available")
    def train(self,ds,**kw)->TrainResult:
        raise RuntimeError("Tinker backend not available")
    def save(self,path:str):
        raise RuntimeError("Tinker backend not available")
