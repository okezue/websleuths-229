import os
from datasets import Dataset
from wm.train.hf import HFBack
from wm.cfg import TrainCfg,DreamCfg,MixCfg
from wm.dream.core import Dreamer

def _ds():
    return Dataset.from_dict({"text":["hello world test data sequence "*8]*30})

def test_train_loop_no_dream(tiny_model,tiny_tok,tmp_dir):
    cfg=TrainCfg(backend="hf",epochs=1,bs=2,max_steps=6,lr=1e-3,
        warmup=1,bf16=False,fp16=False,grad_acc=1)
    b=HFBack(cfg)
    b.setup_from_model(tiny_model,tiny_tok,tmp_dir)
    r=b.train(_ds())
    assert r.steps==6
    assert r.loss>0
    assert r.dream_loss is None

def test_train_loop_dream_on_episode_batch(tiny_model,tiny_tok,tmp_dir):
    cfg=TrainCfg(backend="hf",epochs=1,bs=2,max_steps=6,lr=1e-3,
        warmup=1,bf16=False,fp16=False,grad_acc=1)
    b=HFBack(cfg)
    b.setup_from_model(tiny_model,tiny_tok,tmp_dir)
    d=Dreamer(DreamCfg(mode="logits_kl",weight=0.1),MixCfg(sched="per_step"))
    r=b.train(_ds(),dreamer=d)
    assert r.steps==6
    assert r.dream_loss is not None

def test_train_loop_dream_with_pdream(tiny_model,tiny_tok,tmp_dir):
    pdream=["General knowledge question about science."]*10
    cfg=TrainCfg(backend="hf",epochs=1,bs=2,max_steps=6,lr=1e-3,
        warmup=1,bf16=False,fp16=False,grad_acc=1,
        dream=DreamCfg(n_dream=2,max_len=32))
    b=HFBack(cfg)
    b.setup_from_model(tiny_model,tiny_tok,tmp_dir)
    d=Dreamer(DreamCfg(mode="logits_kl",weight=0.1),MixCfg(sched="per_step"))
    r=b.train(_ds(),dreamer=d,dream_prompts=pdream)
    assert r.steps==6
    assert r.dream_loss is not None

def test_train_save_load(tiny_model,tiny_tok,tmp_dir):
    cfg=TrainCfg(backend="hf",epochs=1,bs=2,max_steps=4,lr=1e-3,
        warmup=1,bf16=False,fp16=False,grad_acc=1)
    b=HFBack(cfg)
    b.setup_from_model(tiny_model,tiny_tok,tmp_dir)
    b.train(_ds())
    sp=os.path.join(tmp_dir,"chk")
    b.save(sp)
    assert os.path.isdir(sp)
