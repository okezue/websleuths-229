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
    history:list[dict[str,float]]=field(default_factory=list)

@dataclass
class EvalReport:
    ppl:float
    acc:float
    qa_f1:float=0.0
    probe_scores:dict[str,float]=field(default_factory=dict)
    extras:dict[str,Any]=field(default_factory=dict)

@dataclass
class VersionInfo:
    vid:str
    parent:str|None
    ts:datetime=field(default_factory=datetime.utcnow)
    metrics:dict[str,float]=field(default_factory=dict)
    path:str=""

@dataclass
class Probe:
    qid:str
    prompt:str
    gold:str
    eid:str
    kind:str

@dataclass
class GuardResult:
    accepted:bool
    u_exp:float=0.0
    u_act:float=0.0
    ci_lo:float=0.0
    ci_hi:float=0.0
    anchor_delta:float=0.0
    drift_kl:float=0.0
    reason:str=""
    phase:str=""

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

@dataclass
class Claim:
    cid:str
    text:str
    eid:str=""
    chunk_idx:int=0
    entities:list[str]=field(default_factory=list)
    confidence:float=1.0

@dataclass
class Entity:
    nid:str
    name:str

@dataclass
class Community:
    coid:str
    label:str
    members:list[str]=field(default_factory=list)
    centroid:dict[str,float]=field(default_factory=dict)
    parameterized:bool=False

@dataclass
class SearchResult:
    topic:str
    claims:list[Claim]=field(default_factory=list)
    entities:list[Entity]=field(default_factory=list)
    communities:list[Community]=field(default_factory=list)
    chunks:list[Chunk]=field(default_factory=list)
    rounds:int=0
    sources:list[str]=field(default_factory=list)

@dataclass
class SearchGateResult:
    should_search:bool
    score:float=0.0
    uncertainty:float=0.0
    freshness:float=0.0
    param_match:float=0.0
    reason:str=""

@dataclass
class UpdateGateResult:
    should_update:bool
    readiness:float=0.0
    novelty:float=0.0
    reason:str=""

@dataclass
class PaceState:
    lam_web:float=0.0
    lam_ft:float=0.0
    tau_search:float=0.5
    tau_ready:float=0.6
    web_calls:int=0
    ft_calls:int=0
