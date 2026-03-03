from __future__ import annotations
from typing import Literal
from pydantic import BaseModel,Field

class IngestCfg(BaseModel):
    src:str="stub"
    max_ep:int=100
    timeout:float=30.0

class ChunkCfg(BaseModel):
    max_tok:int=512
    overlap:int=64
    enc:str="cl100k_base"

class DatasetCfg(BaseModel):
    recipe:Literal["cpt","ext_sft","cited_qa"]="cpt"
    max_len:int=1024
    test_frac:float=0.1
    seed:int=42

class LoraCfg(BaseModel):
    r:int=16
    alpha:int=32
    dropout:float=0.05
    modules:list[str]=Field(default_factory=lambda:["q_proj","v_proj"])

class GateCfg(BaseModel):
    min_sources:int=2
    min_consistency:float=0.3
    uncertainty_thresh:float=0.7
    enabled:bool=True

class DreamCfg(BaseModel):
    mode:Literal["logits_kl","sampled"]="logits_kl"
    temp:float=2.0
    weight:float=0.5
    n_samples:int=16
    n_dream:int=4
    max_len:int=128

class TrainCfg(BaseModel):
    backend:Literal["hf","neuron","tinker"]="hf"
    base_model:str="meta-llama/Llama-3.2-1B"
    epochs:int=3
    bs:int=4
    lr:float=2e-4
    grad_acc:int=4
    max_steps:int=-1
    warmup:int=50
    fp16:bool=False
    bf16:bool=True
    lora:LoraCfg=Field(default_factory=LoraCfg)
    dream:DreamCfg=Field(default_factory=DreamCfg)

class MixCfg(BaseModel):
    sched:Literal["per_step","interleave","two_phase"]="per_step"
    phase_ratio:float=0.5

class EATRDCfg(BaseModel):
    eps_min:float=0.01
    alpha:float=0.5
    rho:float=0.01
    lam_init:float=1.0
class DPMUCfg(BaseModel):
    n_dream_grads:int=1
class EABSSCCfg(BaseModel):
    consolidation_rank:int=16
    beta:float=0.1
    gamma:float=0.01
    tau:float=1.0
    refine_steps:int=20
class RecipeCfg(BaseModel):
    name:Literal["eatrd","dpmu","eab_ssc"]="eatrd"
    eatrd:EATRDCfg=Field(default_factory=EATRDCfg)
    dpmu:DPMUCfg=Field(default_factory=DPMUCfg)
    eab_ssc:EABSSCCfg=Field(default_factory=EABSSCCfg)
class RegCfg(BaseModel):
    local_dir:str="./registry"
    s3_bucket:str|None=None
    s3_prefix:str="wm/"

class CertCfg(BaseModel):
    max_ppl_delta:float=5.0
    min_acc:float=0.3
    max_drift:float=0.1
    n_bootstrap:int=1000
    ci_alpha:float=0.05
    sprt_delta:float=0.05
    sprt_alpha:float=0.05
    sprt_beta:float=0.1

class GuardCfg(BaseModel):
    u_exp_thresh:float=0.0
    max_anchor_delta:float=0.5
    ci_alpha:float=0.05
    n_bootstrap:int=1000
    enabled:bool=True

class SearchCfg(BaseModel):
    max_rounds:int=3
    queries_per_round:int=3
    min_claims:int=5
    marginal_thresh:float=0.1
    mmr_lambda:float=0.7
    mmr_k:int=20
    exa_api_key:str|None=None
    model_query_gen:bool=True
    query_temp:float=0.9

class GraphCfg(BaseModel):
    db_path:str="./wm_graph.db"
    dist_thresh:float=0.5

class SearchGateCfg(BaseModel):
    a:float=0.4
    b:float=0.3
    c:float=0.3
    tau_search:float=0.5
    enabled:bool=True

class UpdateGateCfg(BaseModel):
    a1:float=0.3
    a2:float=0.2
    a3:float=0.3
    a4:float=0.2
    tau_ready:float=0.5
    tau_novel:float=0.3
    enabled:bool=True

class PaceCfg(BaseModel):
    web_budget:int=50
    ft_budget:int=10
    eta:float=0.1

class DomainBenchCfg(BaseModel):
    domains:list[str]=Field(default_factory=lambda:["finance","legal","chemistry"])
    topics:dict[str,str]=Field(default_factory=lambda:{
        "finance":"financial ratio analysis earnings reports",
        "legal":"legal precedent constitutional law",
        "chemistry":"organic chemistry reaction mechanisms",
    })
    mmlu_n:int=50
    probes_per_domain:int=10

class WMCfg(BaseModel):
    ingest:IngestCfg=Field(default_factory=IngestCfg)
    chunk:ChunkCfg=Field(default_factory=ChunkCfg)
    dataset:DatasetCfg=Field(default_factory=DatasetCfg)
    train:TrainCfg=Field(default_factory=TrainCfg)
    mix:MixCfg=Field(default_factory=MixCfg)
    gate:GateCfg=Field(default_factory=GateCfg)
    recipe:RecipeCfg=Field(default_factory=RecipeCfg)
    reg:RegCfg=Field(default_factory=RegCfg)
    cert:CertCfg=Field(default_factory=CertCfg)
    guard:GuardCfg=Field(default_factory=GuardCfg)
    search:SearchCfg=Field(default_factory=SearchCfg)
    graph:GraphCfg=Field(default_factory=GraphCfg)
    search_gate:SearchGateCfg=Field(default_factory=SearchGateCfg)
    update_gate:UpdateGateCfg=Field(default_factory=UpdateGateCfg)
    pace:PaceCfg=Field(default_factory=PaceCfg)
    domain_bench:DomainBenchCfg=Field(default_factory=DomainBenchCfg)
