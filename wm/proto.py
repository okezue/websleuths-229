from __future__ import annotations
from typing import Protocol,runtime_checkable
from wm.types import Episode,Chunk,TrainResult,EvalReport
import torch

@runtime_checkable
class WebSrc(Protocol):
    def fetch(self,query:str,n:int)->list[Episode]:...

@runtime_checkable
class TrainBack(Protocol):
    def setup(self,model_name:str,lora_cfg:object,out_dir:str)->None:...
    def train(self,ds:object,**kw:object)->TrainResult:...
    def save(self,path:str)->None:...

@runtime_checkable
class DreamMode(Protocol):
    def loss(self,student_logits:torch.Tensor,teacher_logits:torch.Tensor)->torch.Tensor:...

@runtime_checkable
class MixSched(Protocol):
    def should_dream(self,step:int,total:int)->bool:...
