import pytest
from wm.gate.update_gate import UpdateGate,_readiness
from wm.cfg import UpdateGateCfg
from wm.types import SearchResult,Claim

def _make_sr(n_claims=5,n_sources=3):
    claims=[Claim(cid=f"c{i}",text=f"Claim number {i} about important topic.",
                  entities=[f"Entity{i}"],confidence=0.8)
            for i in range(n_claims)]
    sources=[f"https://source{i}.com/article" for i in range(n_sources)]
    return SearchResult(topic="test",claims=claims,sources=sources)

def test_update_gate_disabled():
    cfg=UpdateGateCfg(enabled=False)
    ug=UpdateGate(cfg)
    sr=_make_sr(1,1)
    r=ug.check(sr)
    assert r.should_update

def test_readiness_increases_with_sources():
    cfg=UpdateGateCfg()
    r1=_readiness(_make_sr(3,1),cfg)
    r2=_readiness(_make_sr(3,5),cfg)
    assert r2>r1

def test_update_gate_low_readiness():
    cfg=UpdateGateCfg(tau_ready=0.99,tau_novel=0.0)
    ug=UpdateGate(cfg)
    sr=_make_sr(1,1)
    r=ug.check(sr)
    assert not r.should_update

def test_update_gate_sufficient():
    cfg=UpdateGateCfg(tau_ready=0.3,tau_novel=0.0)
    ug=UpdateGate(cfg)
    sr=_make_sr(10,5)
    r=ug.check(sr)
    assert r.should_update

def test_update_gate_with_model(tiny_model,tiny_tok):
    cfg=UpdateGateCfg(tau_ready=0.3,tau_novel=0.0)
    ug=UpdateGate(cfg)
    sr=_make_sr(5,3)
    r=ug.check(sr,model=tiny_model,tok=tiny_tok)
    assert isinstance(r.novelty,float)
    assert r.novelty>0
