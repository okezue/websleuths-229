from __future__ import annotations
from dataclasses import dataclass,field
from datetime import datetime
from hashlib import sha256
from typing import Any

@dataclass
class Episode:
    url:str
    title:str
    body:str
    ts:datetime=field(default_factory=datetime.utcnow)
    meta:dict[str,Any]=field(default_factory=dict)
    authority:float=0.0
    topic:str=""
    @property
    def eid(self)->str:
        return sha256(self.url.encode()).hexdigest()[:16]

@dataclass
class GateResult:
    accept:bool
    n_sources:int=0
    consistency:float=0.0
    uncertainty:float=0.0
    reason:str=""

@dataclass
class Chunk:
    eid:str
    idx:int
    text:str
    n_tok:int=0
    authority:float=1.0

@dataclass
class TrainResult:
    loss:float
    steps:int
    lr:float
    dream_loss:float|None=None
    extras:dict[str,Any]=field(default_factory=dict)

@dataclass
class EvalReport:
    ppl:float
    acc:float
    qa_f1:float=0.0
    extras:dict[str,Any]=field(default_factory=dict)

@dataclass
class VersionInfo:
    vid:str
    parent:str|None
    ts:datetime=field(default_factory=datetime.utcnow)
    metrics:dict[str,float]=field(default_factory=dict)
    path:str=""

@dataclass
class CheckResult:
    name:str
    passed:bool
    val:float=0.0
    thresh:float=0.0

@dataclass
class CertResult:
    passed:bool
    checks:list[CheckResult]=field(default_factory=list)
