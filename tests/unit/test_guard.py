import torch,copy
from wm.guard.gain import expected_gain
from wm.guard.drift import drift_kl
from wm.guard.orchestrator import UpdateGuard
from wm.cfg import GuardCfg
from wm.types import Chunk

def test_expected_gain_positive():
    g_k=torch.tensor([1.0,2.0,3.0])
    g_a=torch.tensor([1.0,1.0,1.0])
    u=expected_gain(g_k,g_a,alpha=0.5)
    assert u>0
    assert isinstance(u,float)

def test_expected_gain_varies_alpha():
    g_k=torch.tensor([1.0,-1.0])
    g_a=torch.tensor([1.0,2.0])
    u1=expected_gain(g_k,g_a,alpha=0.0)
    u2=expected_gain(g_k,g_a,alpha=1.0)
    assert u1!=u2

def test_expected_gain_aligned():
    g_k=torch.tensor([1.0,1.0])
    g_a=torch.tensor([1.0,1.0])
    u=expected_gain(g_k,g_a,alpha=0.5)
    assert u==g_k.dot(g_k).item()

def test_drift_kl_identical(tiny_model,tiny_tok):
    m2=copy.deepcopy(tiny_model)
    d=drift_kl(tiny_model,m2,tiny_tok,["Hello world."],max_len=32)
    assert d<0.01

def test_drift_kl_type(tiny_model,tiny_tok):
    m2=copy.deepcopy(tiny_model)
    d=drift_kl(tiny_model,m2,tiny_tok,["Test.","Data."],max_len=32)
    assert isinstance(d,float)

def _chunks():
    return [
        Chunk(eid="a"*16,idx=0,
            text="Albert Einstein developed relativity in 1905. This changed physics forever.",
            n_tok=15,authority=0.9),
    ]

def test_guard_accept(tiny_model,tiny_tok):
    cfg=GuardCfg(u_exp_thresh=-999,max_anchor_delta=999,n_bootstrap=50)
    guard=UpdateGuard(cfg,tiny_tok,anchors=["Hello."])
    def noop(m,d,dp):pass
    r=guard.guard(tiny_model,[],noop,None,[])
    assert r.accepted
    assert r.phase=="done"

def test_guard_reject_uexp(tiny_model,tiny_tok):
    cfg=GuardCfg(u_exp_thresh=999,n_bootstrap=50)
    guard=UpdateGuard(cfg,tiny_tok,anchors=["Hello."])
    def noop(m,d,dp):pass
    g_k=torch.tensor([0.01])
    g_a=torch.tensor([0.01])
    r=guard.guard(tiny_model,_chunks(),noop,None,[],g_k=g_k,g_a=g_a)
    assert not r.accepted
    assert r.phase=="pre_ft"
