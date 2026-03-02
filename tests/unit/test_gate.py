from datetime import datetime
from wm.types import Episode
from wm.gate import EpisodeGate,pagerank_authority
from wm.cfg import GateCfg

def _eps_multi_domain(n=5):
    shared="The suspect was seen near the crime scene in downtown. Police investigated the area thoroughly. Evidence was collected from multiple witnesses and forensic sources."
    eps=[]
    doms=["cnn.com","bbc.co.uk","nytimes.com","reuters.com","apnews.com"]
    for i in range(n):
        eps.append(Episode(
            url=f"https://{doms[i%len(doms)]}/article-{i}",
            title=f"Article {i}",
            body=f"{shared} Additional detail number {i}. "*5,
            ts=datetime(2025,1,1),
        ))
    return eps

def _eps_single_domain(n=3):
    return [Episode(
        url=f"https://example.com/page-{i}",
        title=f"Page {i}",
        body=f"Content about topic {i}. "*10,
        ts=datetime(2025,1,1),
    ) for i in range(n)]

def test_gate_accept_diverse():
    g=EpisodeGate(GateCfg(min_sources=2,min_consistency=0.1,uncertainty_thresh=0.0))
    r=g.check(_eps_multi_domain(4))
    assert r.accept
    assert r.n_sources>=2

def test_gate_reject_single_source():
    g=EpisodeGate(GateCfg(min_sources=3))
    r=g.check(_eps_single_domain(5))
    assert not r.accept
    assert "sources" in r.reason

def test_gate_consistency():
    g=EpisodeGate(GateCfg(min_sources=1,min_consistency=0.0))
    r=g.check(_eps_multi_domain(3))
    assert r.accept
    assert r.consistency>0

def test_gate_reject_low_consistency():
    unrelated=[
        Episode(url=f"https://dom{i}.com/x",title="t",
            body=f"completely different topic number {i} with unique words set{i} "*20,
            ts=datetime(2025,1,1))
        for i in range(3)
    ]
    g=EpisodeGate(GateCfg(min_sources=1,min_consistency=0.99))
    r=g.check(unrelated)
    assert not r.accept
    assert "consistency" in r.reason

def test_gate_disabled():
    g=EpisodeGate(GateCfg(enabled=False))
    r=g.check([])
    assert r.accept

def test_gate_with_model_uncertainty(tiny_model,tiny_tok):
    g=EpisodeGate(GateCfg(min_sources=1,min_consistency=0.0,uncertainty_thresh=0.0))
    eps=_eps_multi_domain(3)
    r=g.check(eps,model=tiny_model,tok=tiny_tok)
    assert r.accept
    assert r.uncertainty>0

def test_exa_authority():
    eps=_eps_multi_domain(5)
    ranked=pagerank_authority(eps)
    assert len(ranked)==5
    assert all(e.authority>=0 for e in ranked)
    assert ranked[0].authority>=ranked[-1].authority

def test_exa_authority_single():
    eps=[Episode(url="https://x.com/1",title="t",body="b",authority=0.8)]
    ranked=pagerank_authority(eps)
    assert ranked[0].authority>0

def test_exa_authority_empty():
    assert pagerank_authority([])==[]

def test_gate_reject_low_uncertainty(tiny_model,tiny_tok):
    eps=_eps_multi_domain(3)
    g=EpisodeGate(GateCfg(min_sources=1,min_consistency=0.0,uncertainty_thresh=999.0))
    r=g.check(eps,model=tiny_model,tok=tiny_tok)
    assert not r.accept
    assert "uncertainty" in r.reason
