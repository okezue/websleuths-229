import torch
from wm.eval.anchor import AnchorEval

def test_nll(tiny_model,tiny_tok):
    ae=AnchorEval(tiny_model,tiny_tok,anchors=["Hello world.","Test data."])
    v=ae.nll()
    assert isinstance(v,float)
    assert v>0

def test_per_anchor_nll(tiny_model,tiny_tok):
    ae=AnchorEval(tiny_model,tiny_tok,anchors=["Hello.","World.","Test."])
    vals=ae.per_anchor_nll()
    assert len(vals)==3
    assert all(v>0 for v in vals)

def test_delta_nll(tiny_model,tiny_tok):
    ae=AnchorEval(tiny_model,tiny_tok,anchors=["Hello world."])
    pre=ae.nll()
    d=ae.delta_nll(pre)
    assert abs(d)<1e-5

def test_default_anchors(tiny_model,tiny_tok):
    ae=AnchorEval(tiny_model,tiny_tok)
    v=ae.nll()
    assert v>0
