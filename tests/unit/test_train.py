import os
import pytest
from datasets import Dataset
from wm.train import make_back
from wm.train.hf import HFBack,DreamBuffer
from wm.train.neuron import NeuronBack
from wm.train.tinker import TinkerBack
from wm.cfg import TrainCfg
from wm.dream.core import Dreamer
from wm.cfg import DreamCfg,MixCfg

def _ds():
    return Dataset.from_dict({"text":["hello world this is test "*10]*20})

_DREAM_PROMPTS=[
    "What is the capital of France?",
    "Explain how photosynthesis works.",
    "Who wrote Romeo and Juliet?",
    "What is the speed of light?",
    "Describe the water cycle.",
]

def test_make_back_hf():
    b=make_back(TrainCfg(backend="hf"))
    assert isinstance(b,HFBack)

def test_make_back_neuron():
    b=make_back(TrainCfg(backend="neuron"))
    assert isinstance(b,NeuronBack)
    assert not b.available

def test_neuron_raises():
    b=NeuronBack()
    with pytest.raises(RuntimeError):
        b.setup("x",None,"out")
    with pytest.raises(RuntimeError):
        b.train(None)
    with pytest.raises(RuntimeError):
        b.save("x")

def test_make_back_tinker():
    b=make_back(TrainCfg(backend="tinker"))
    assert isinstance(b,TinkerBack)
    assert not b.available

def test_tinker_raises():
    b=TinkerBack()
    with pytest.raises(RuntimeError):
        b.setup("x",None,"out")
    with pytest.raises(RuntimeError):
        b.train(None)
    with pytest.raises(RuntimeError):
        b.save("x")

def test_hf_train(tiny_model,tiny_tok,tmp_dir):
    cfg=TrainCfg(backend="hf",epochs=1,bs=2,max_steps=4,lr=1e-3,
        warmup=1,bf16=False,fp16=False,grad_acc=1)
    b=HFBack(cfg)
    b.setup_from_model(tiny_model,tiny_tok,tmp_dir)
    res=b.train(_ds())
    assert res.steps==4
    assert res.loss>0

def test_hf_train_dream_no_buffer(tiny_model,tiny_tok,tmp_dir):
    cfg=TrainCfg(backend="hf",epochs=1,bs=2,max_steps=4,lr=1e-3,
        warmup=1,bf16=False,fp16=False,grad_acc=1)
    b=HFBack(cfg)
    b.setup_from_model(tiny_model,tiny_tok,tmp_dir)
    d=Dreamer(DreamCfg(mode="logits_kl",weight=0.1),MixCfg(sched="per_step"))
    res=b.train(_ds(),dreamer=d)
    assert res.dream_loss is not None

def test_hf_train_dream_with_pdream(tiny_model,tiny_tok,tmp_dir):
    cfg=TrainCfg(backend="hf",epochs=1,bs=2,max_steps=4,lr=1e-3,
        warmup=1,bf16=False,fp16=False,grad_acc=1,
        dream=DreamCfg(n_dream=2,max_len=32))
    b=HFBack(cfg)
    b.setup_from_model(tiny_model,tiny_tok,tmp_dir)
    d=Dreamer(DreamCfg(mode="logits_kl",weight=0.1),MixCfg(sched="per_step"))
    res=b.train(_ds(),dreamer=d,dream_prompts=_DREAM_PROMPTS)
    assert res.dream_loss is not None

def test_dream_buffer(tiny_tok):
    import torch
    buf=DreamBuffer(_DREAM_PROMPTS,tiny_tok,max_len=32,n=2)
    s=buf.sample(torch.device("cpu"))
    assert s is not None
    assert "input_ids" in s
    assert s["input_ids"].shape[0]==2

def test_dream_buffer_empty(tiny_tok):
    import torch
    buf=DreamBuffer([],tiny_tok,max_len=32,n=2)
    assert buf.sample(torch.device("cpu")) is None

def test_hf_setup_from_pretrained(tiny_model,tiny_tok,tmp_dir):
    mp=os.path.join(tmp_dir,"base_model")
    tiny_model.save_pretrained(mp)
    tiny_tok.save_pretrained(mp)
    cfg=TrainCfg(backend="hf",epochs=1,bs=2,max_steps=2,lr=1e-3,
        warmup=1,bf16=False,fp16=False,grad_acc=1)
    b=HFBack(cfg)
    b.setup(mp,cfg.lora,tmp_dir)
    assert b.model is not None
    assert b.tokenizer is not None

def test_hf_train_sft_format(tiny_model,tiny_tok,tmp_dir):
    ds=Dataset.from_dict({
        "prompt":["What is X?"]*10,
        "completion":["X is Y."]*10,
    })
    cfg=TrainCfg(backend="hf",epochs=1,bs=2,max_steps=4,lr=1e-3,
        warmup=1,bf16=False,fp16=False,grad_acc=1)
    b=HFBack(cfg)
    b.setup_from_model(tiny_model,tiny_tok,tmp_dir)
    res=b.train(ds)
    assert res.steps==4
    assert res.loss>0

def test_hf_save(tiny_model,tiny_tok,tmp_dir):
    cfg=TrainCfg(backend="hf",epochs=1,bs=2,max_steps=2,lr=1e-3,
        warmup=1,bf16=False,fp16=False,grad_acc=1)
    b=HFBack(cfg)
    b.setup_from_model(tiny_model,tiny_tok,tmp_dir)
    b.train(_ds())
    sp=os.path.join(tmp_dir,"saved")
    b.save(sp)
    assert os.path.exists(sp)
