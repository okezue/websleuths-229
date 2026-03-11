import torch
from wm.dream import LogitsKL,SampledKL,Dreamer,PerStep,Interleave,TwoPhase
from wm.cfg import DreamCfg,MixCfg
from wm.proto import DreamMode,MixSched

def _logits(bs=2,seq=4,v=32):
    return torch.randn(bs,seq,v)

def test_logits_kl():
    m=LogitsKL(temp=2.0)
    l=m.loss(_logits(),_logits())
    assert l.shape==()
    assert l.item()>=0

def test_logits_kl_identical():
    x=_logits()
    m=LogitsKL(temp=2.0)
    l=m.loss(x,x)
    assert l.item()>=0
    assert l.item()<0.01

def test_sampled_kl():
    m=SampledKL(temp=2.0,n_samples=8)
    l=m.loss(_logits(),_logits())
    assert l.shape==()

def test_per_step():
    s=PerStep()
    assert all(s.should_dream(i,10) for i in range(10))

def test_interleave():
    s=Interleave()
    assert not s.should_dream(0,10)
    assert s.should_dream(1,10)
    assert not s.should_dream(2,10)

def test_two_phase():
    s=TwoPhase(ratio=0.5)
    assert not s.should_dream(0,10)
    assert not s.should_dream(4,10)
    assert s.should_dream(5,10)
    assert s.should_dream(9,10)

def test_dreamer_per_step():
    d=Dreamer(DreamCfg(mode="logits_kl"),MixCfg(sched="per_step"))
    l=d.step(0,10,_logits(),_logits())
    assert l is not None

def test_dreamer_interleave():
    d=Dreamer(DreamCfg(mode="sampled"),MixCfg(sched="interleave"))
    assert d.step(0,10,_logits(),_logits()) is None
    assert d.step(1,10,_logits(),_logits()) is not None

def test_dreamer_two_phase():
    d=Dreamer(DreamCfg(),MixCfg(sched="two_phase",phase_ratio=0.5))
    assert d.step(2,10,_logits(),_logits()) is None
    assert d.step(7,10,_logits(),_logits()) is not None

def test_protocol_compliance():
    assert isinstance(LogitsKL(),DreamMode)
    assert isinstance(SampledKL(),DreamMode)
    assert isinstance(PerStep(),MixSched)
    assert isinstance(Interleave(),MixSched)
    assert isinstance(TwoPhase(),MixSched)
