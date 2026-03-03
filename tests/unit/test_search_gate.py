import pytest
from wm.gate.search_gate import SearchGate,_freshness,_param_match
from wm.cfg import SearchGateCfg
from wm.types import Community

def test_freshness_positive():
    assert _freshness("latest news 2025")==1.0
    assert _freshness("current events")==1.0
    assert _freshness("breaking news")==1.0

def test_freshness_negative():
    assert _freshness("history of Rome")==0.0

def test_param_match_no_communities():
    assert _param_match("test query",[])==0.0

def test_param_match_matching():
    co=Community(coid="co1",label="finance",members=[],
                 centroid={"finance":0.8,"markets":0.6})
    score=_param_match("finance markets trends",[co])
    assert score>0.3

def test_search_gate_disabled():
    cfg=SearchGateCfg(enabled=False)
    sg=SearchGate(cfg)
    r=sg.check("anything")
    assert r.should_search

def test_search_gate_no_model():
    cfg=SearchGateCfg(a=0.4,b=0.3,c=0.3,tau_search=0.3)
    sg=SearchGate(cfg)
    r=sg.check("latest financial news 2025")
    assert r.uncertainty==1.0
    assert r.freshness==1.0
    assert r.should_search

def test_search_gate_parameterized_community():
    cfg=SearchGateCfg(a=0.0,b=0.0,c=1.0,tau_search=0.0)
    sg=SearchGate(cfg)
    co=Community(coid="co1",label="finance",members=[],
                 centroid={"finance":0.9,"analysis":0.5},parameterized=True)
    r=sg.check("finance analysis",[co])
    assert not r.should_search

def test_search_gate_with_model(tiny_model,tiny_tok):
    cfg=SearchGateCfg(tau_search=0.0)
    sg=SearchGate(cfg)
    r=sg.check("hello world",model=tiny_model,tok=tiny_tok)
    assert isinstance(r.uncertainty,float)
    assert r.should_search
