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
    tinker_api_key:str|None=None
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
    d_targ:float=0.5
    lam_floor:float=0.01
    lam_ceil:float=10.0
    use_pi:bool=True
class DPMUCfg(BaseModel):
    n_dream_grads:int=1
    grad_refresh_k:int=5
    grad_ema_decay:float=0.9
class DreamBankCfg(BaseModel):
    enabled:bool=True
    bucket_weights:dict[str,float]=Field(default_factory=lambda:{"if_canary":0.20,"creative":0.15,"ood_noise":0.15,"reasoning":0.15,"anchor":0.20,"episode":0.15})
    bucket_temps:dict[str,float]=Field(default_factory=lambda:{"if_canary":1.0,"creative":2.0,"ood_noise":1.5,"reasoning":1.0,"anchor":1.5,"episode":1.5})
    hdm_pool:int=50
    hdm_topk:int=10
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

class DistillCfg(BaseModel):
    enabled:bool=True
    n_questions:int=20
    mu_init:float=0.5
    mu_floor:float=0.05
    mu_ceil:float=2.0
    openai_api_key:str|None=None
    gpt_model:str="gpt-5.4"
    claude_thinking:bool=True
    claude_think_budget:int=10000
class NeurogenesisCfg(BaseModel):
    enabled:bool=True
    max_adapters:int=20
    spawn_loss_thresh:float=2.0
    spawn_dream_thresh:float=1.5
    spawn_fail_thresh:int=2
    rank_step:int=8
    max_rank:int=128
    ortho_weight:float=0.01
class SearchCfg(BaseModel):
    max_rounds:int=5
    queries_per_round:int=6
    min_claims:int=5
    marginal_thresh:float=0.1
    mmr_lambda:float=0.7
    mmr_k:int=50
    exa_api_key:str|None=None
    parallel_api_key:str|None=None
    model_query_gen:bool=True
    query_temp:float=0.9
    round_temps:list[float]=Field(default_factory=lambda:[0.7,0.9,1.0,1.0,1.0])
    res_per_query:int=5
    extraction_backend:Literal["regex","claude"]="claude"
    anthropic_api_key:str|None=None
    claude_model:str="claude-sonnet-4-5-20250929"
    claude_concurrency:int=10
    use_multi_search:bool=True
    procedural_extraction:bool=True

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

class IterCLCfg(BaseModel):
    bench_n:int=100
    mmlu_n:int=50
    steps_per_topic:int=50
    ckpt_dir:str="/tmp/wm_checkpoints"
    recipes:list[str]=Field(default_factory=lambda:["eatrd","dpmu","eab_ssc"])
    min_steps:int=30
    max_steps:int=150
    step_alpha:float=0.5
    step_beta:float=10.0

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
    dream_bank:DreamBankCfg=Field(default_factory=DreamBankCfg)
    iter_cl:IterCLCfg=Field(default_factory=IterCLCfg)
    distill:DistillCfg=Field(default_factory=DistillCfg)
    neurogenesis:NeurogenesisCfg=Field(default_factory=NeurogenesisCfg)
