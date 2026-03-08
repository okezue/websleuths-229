import pytest,torch,copy
from collections import OrderedDict
from wm.recipe.neurogenesis import (
    AdapterSlot,NeurogenesisBank,RankGrowthLoRA,NeurogenesisEATRD)

def test_adapter_slot_defaults():
    s=AdapterSlot(0,OrderedDict())
    assert s.idx==0
    assert s.frozen==False
    assert s.guard_failures==0

def test_bank_snapshot(tiny_model):
    bank=NeurogenesisBank()
    assert bank.n_adapters==0
    slot=bank.snapshot_adapter(tiny_model,domain="finance",topic="test")
    assert bank.n_adapters==1
    assert slot.domain=="finance"
    assert len(slot.sd)>0

def test_bank_freeze(tiny_model):
    bank=NeurogenesisBank()
    bank.snapshot_adapter(tiny_model)
    bank.freeze_adapter(0)
    assert bank.slots[0].frozen==True

def test_bank_freeze_all(tiny_model):
    bank=NeurogenesisBank()
    bank.snapshot_adapter(tiny_model)
    bank.snapshot_adapter(tiny_model)
    bank.freeze_all()
    assert all(s.frozen for s in bank.slots)

def test_bank_restore(tiny_model):
    bank=NeurogenesisBank()
    bank.snapshot_adapter(tiny_model)
    orig={n:p.clone() for n,p in tiny_model.named_parameters() if p.requires_grad}
    for p in tiny_model.parameters():
        if p.requires_grad:
            p.data.add_(torch.randn_like(p)*0.1)
    bank.restore_adapter(tiny_model,0)
    for n,p in tiny_model.named_parameters():
        if n in orig:
            assert torch.allclose(p,orig[n],atol=1e-6)

def test_should_spawn_guard_failures():
    bank=NeurogenesisBank()
    s,r=bank.should_spawn(0.5,0.5,guard_failures=3)
    assert s==True
    assert r=="guard_failures"

def test_should_spawn_high_loss():
    bank=NeurogenesisBank()
    s,r=bank.should_spawn(5.0,0.5,guard_failures=0)
    assert s==True
    assert r=="high_residual_loss"

def test_should_spawn_conflict():
    bank=NeurogenesisBank()
    s,r=bank.should_spawn(1.5,2.0,guard_failures=0)
    assert s==True
    assert r=="conflict"

def test_should_not_spawn():
    bank=NeurogenesisBank()
    s,r=bank.should_spawn(0.5,0.3,guard_failures=0)
    assert s==False

def test_rank_growth_init():
    rg=RankGrowthLoRA(r_step=8,max_total_r=64)
    assert rg.n_chunks==0
    assert rg.total_rank==0

def test_rank_growth_ortho_empty():
    rg=RankGrowthLoRA()
    from transformers import AutoModelForCausalLM,LlamaConfig
    cfg=LlamaConfig(vocab_size=256,hidden_size=64,intermediate_size=128,
                    num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2)
    m=AutoModelForCausalLM.from_config(cfg)
    pen=rg.ortho_penalty(m)
    assert pen.item()==0.0

def test_neurogenesis_eatrd_init():
    r=NeurogenesisEATRD(lr=1e-3,max_steps=5)
    assert r.lr==1e-3
    assert r.ms==5
    assert r.rg is not None
