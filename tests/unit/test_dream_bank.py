import torch
from wm.dream.bank import DreamBank,BUCKETS
from wm.types import Community,Claim

def test_init_empty(tiny_tok):
    b=DreamBank(tiny_tok)
    for k in BUCKETS:
        assert b.bucket_sizes()[k]==0

def test_seed_fills_buckets(tiny_tok):
    b=DreamBank(tiny_tok)
    b.seed()
    sz=b.bucket_sizes()
    assert sz["if_canary"]>=15
    assert sz["creative"]>=15
    assert sz["reasoning"]>=15
    assert sz["anchor"]>=10
    assert sz["ood_noise"]>=10
    assert sz["episode"]==0

def test_sample_returns_tensors(tiny_tok):
    b=DreamBank(tiny_tok,n=2)
    b.seed()
    dev=torch.device("cpu")
    out=b.sample(dev)
    assert out is not None
    assert "input_ids" in out
    assert out["input_ids"].shape[0]==2

def test_sample_with_temps(tiny_tok):
    b=DreamBank(tiny_tok,n=3)
    b.seed()
    dev=torch.device("cpu")
    res=b.sample_with_temps(dev)
    assert res is not None
    enc,temps=res
    assert len(temps)==3
    assert all(t>0 for t in temps)
    assert enc["input_ids"].shape[0]==3

def test_add_episode(tiny_tok):
    b=DreamBank(tiny_tok)
    b.seed()
    comms=[Community(coid="c1",label="test topic")]
    claims=[Claim(cid="cl1",text="test claim")]
    b.add_episode(comms,claims)
    assert b.bucket_sizes()["episode"]>0

def test_to_flat(tiny_tok):
    b=DreamBank(tiny_tok)
    b.seed()
    flat=b.to_flat()
    assert isinstance(flat,list)
    assert len(flat)>50

def test_from_flat(tiny_tok):
    prompts=["What is gravity?","How does rain form?"]
    b=DreamBank.from_flat(prompts,tiny_tok)
    assert b.bucket_sizes()["anchor"]==2
    dev=torch.device("cpu")
    out=b.sample(dev)
    assert out is not None

def test_sample_pool(tiny_tok):
    b=DreamBank(tiny_tok)
    b.seed()
    pairs=b.sample_pool(10)
    assert len(pairs)==10
    for txt,bk in pairs:
        assert isinstance(txt,str)
        assert bk in BUCKETS

def test_empty_sample_none(tiny_tok):
    b=DreamBank(tiny_tok)
    assert b.sample(torch.device("cpu")) is None
    assert b.sample_with_temps(torch.device("cpu")) is None
