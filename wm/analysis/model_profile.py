from __future__ import annotations
import logging,json,os,subprocess,tempfile
from dataclasses import dataclass,field

log=logging.getLogger(__name__)

@dataclass
class ProfileResult:
    model_id:str=""
    hardware:str=""
    total_params:int=0
    trainable_params:int=0
    decode_time_ms:float=0.0
    prefill_time_ms:float=0.0
    memory_gb:float=0.0
    bound:str=""
    per_layer:list[dict]=field(default_factory=list)
    raw:dict=field(default_factory=dict)

class ModelProfiler:
    def __init__(self,hardware:str="nvidia_A10G"):
        self._hw=hardware
        self._viewer=None
    def _try_load_viewer(self):
        if self._viewer is not None:
            return True
        try:
            from model_analyzer import ModelAnalyzer
            self._viewer=ModelAnalyzer
            return True
        except ImportError:
            log.debug("LLM-Viewer not installed")
            return False
    def profile_model(self,model_id:str,seqlen:int=2048,
                       batchsize:int=1,w_bit:int=4)->ProfileResult:
        if not self._try_load_viewer():
            return self._profile_torch(model_id)
        try:
            analyzer=self._viewer(model_id=model_id,hardware=self._hw)
            res=analyzer.analyze(seqlen=seqlen,batchsize=batchsize,
                                  w_bit=w_bit)
            return ProfileResult(
                model_id=model_id,hardware=self._hw,
                decode_time_ms=res.get("decode_total_time",0)*1000,
                prefill_time_ms=res.get("prefill_total_time",0)*1000,
                memory_gb=res.get("total_memory",0)/1e9,
                bound=res.get("bound","unknown"),
                raw=res)
        except Exception as e:
            log.warning("LLM-Viewer analysis failed: %s",e)
            return self._profile_torch(model_id)
    def _profile_torch(self,model_id:str)->ProfileResult:
        try:
            from transformers import AutoConfig
            cfg=AutoConfig.from_pretrained(model_id,trust_remote_code=True)
            hs=getattr(cfg,"hidden_size",0)
            nl=getattr(cfg,"num_hidden_layers",0)
            vs=getattr(cfg,"vocab_size",0)
            ims=getattr(cfg,"intermediate_size",0)
            nah=getattr(cfg,"num_attention_heads",0)
            total=0
            if hs and nl and vs:
                attn=4*hs*hs
                ffn=3*hs*ims if ims else 8*hs*hs
                emb=vs*hs
                total=nl*(attn+ffn)+2*emb
            return ProfileResult(model_id=model_id,total_params=total,
                                  hardware=self._hw)
        except Exception as e:
            log.warning("torch profile failed: %s",e)
            return ProfileResult(model_id=model_id)
    def profile_live_model(self,model,tok=None)->ProfileResult:
        tp=sum(p.numel() for p in model.parameters())
        trp=sum(p.numel() for p in model.parameters() if p.requires_grad)
        mem=0.0
        try:
            import torch
            if torch.cuda.is_available():
                mem=torch.cuda.max_memory_allocated()/1e9
        except Exception:
            pass
        mid=getattr(model,"name_or_path","") or getattr(model.config,"_name_or_path","unknown")
        return ProfileResult(model_id=mid,total_params=tp,trainable_params=trp,
                              memory_gb=mem,hardware=self._hw)
    def compare_profiles(self,before:ProfileResult,after:ProfileResult)->dict:
        return {
            "model":after.model_id,
            "param_delta":after.total_params-before.total_params,
            "trainable_delta":after.trainable_params-before.trainable_params,
            "memory_delta_gb":after.memory_gb-before.memory_gb,
        }
