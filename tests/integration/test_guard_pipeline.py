import copy,torch
from wm.guard.orchestrator import UpdateGuard
from wm.cfg import GuardCfg
from wm.types import Chunk

def _chunks():
    return [
        Chunk(eid="a"*16,idx=0,
            text="Albert Einstein developed relativity in 1905. This changed modern physics forever.",
            n_tok=15,authority=0.9),
        Chunk(eid="b"*16,idx=1,
            text="The Pacific Ocean covers 165250000 square kilometers. It is the largest ocean.",
            n_tok=15,authority=0.7),
    ]

def test_full_guard_pipeline(tiny_model,tiny_tok):
    cfg=GuardCfg(u_exp_thresh=-999,max_anchor_delta=999,n_bootstrap=50)
    guard=UpdateGuard(cfg,tiny_tok,anchors=["Hello world.","Test data."])
    snap_before={k:v.clone() for k,v in tiny_model.state_dict().items()}
    def train_fn(m,d,dp):
        x=torch.randint(0,256,(1,8))
        out=m(x,labels=x)
        out.loss.backward()
        with torch.no_grad():
            for p in m.parameters():
                if p.requires_grad and p.grad is not None:
                    p.data-=0.001*p.grad
        m.zero_grad()
    r=guard.guard(tiny_model,_chunks(),train_fn,None,[])
    assert hasattr(r,"accepted")
    assert hasattr(r,"u_exp")
    assert hasattr(r,"ci_lo")
    assert hasattr(r,"ci_hi")
    assert hasattr(r,"anchor_delta")
    assert hasattr(r,"drift_kl")
    assert hasattr(r,"reason")
    assert hasattr(r,"phase")
    assert isinstance(r.anchor_delta,float)

def test_guard_rollback_on_reject(tiny_model,tiny_tok):
    cfg=GuardCfg(u_exp_thresh=9999,n_bootstrap=50)
    guard=UpdateGuard(cfg,tiny_tok,anchors=["Hello."])
    snap={k:v.clone() for k,v in tiny_model.state_dict().items()}
    g_k=torch.tensor([0.001])
    g_a=torch.tensor([0.001])
    def noop(m,d,dp):pass
    r=guard.guard(tiny_model,[],noop,None,[],g_k=g_k,g_a=g_a)
    assert not r.accepted
    for k in snap:
        assert torch.equal(tiny_model.state_dict()[k],snap[k])
