import os
from wm.ingest import EpisodeIngestor,StubSrc
from wm.store import EpisodeStore
from wm.chunk import Chunker
from wm.dataset import DatasetBuilder
from wm.train.hf import HFBack
from wm.eval import Evaluator
from wm.registry import ModelRegistry
from wm.cert import Certifier
from wm.dream.core import Dreamer
from wm.gate import EpisodeGate,pagerank_authority
from wm.cfg import (
    ChunkCfg,DatasetCfg,TrainCfg,DreamCfg,MixCfg,RegCfg,CertCfg,GateCfg,
)

_P_DREAM=[
    "What is the capital of France?",
    "Explain gravity in simple terms.",
    "Who discovered penicillin?",
    "What causes rain?",
    "How does a computer work?",
]

def test_full_pipeline(tiny_model,tiny_tok,tmp_dir):
    eps=EpisodeIngestor(StubSrc()).ingest("cold case",5)
    eps=pagerank_authority(eps)
    assert eps[0].authority>0

    gate=EpisodeGate(GateCfg(min_sources=1,min_consistency=0.0,uncertainty_thresh=0.0))
    gr=gate.check(eps,model=tiny_model,tok=tiny_tok)
    assert gr.accept

    store=EpisodeStore(os.path.join(tmp_dir,"ep.db"))
    n=store.put_many(eps)
    assert n==5

    chunks=Chunker(ChunkCfg(max_tok=64,overlap=8)).chunk_many(store.all())
    assert len(chunks)>0

    ds=DatasetBuilder(DatasetCfg(recipe="cpt")).build(chunks)
    assert len(ds)>0

    cfg=TrainCfg(backend="hf",epochs=1,bs=2,max_steps=4,lr=1e-3,
        warmup=1,bf16=False,fp16=False,grad_acc=1,
        dream=DreamCfg(n_dream=2,max_len=32))
    back=HFBack(cfg)
    back.setup_from_model(tiny_model,tiny_tok,tmp_dir)
    dreamer=Dreamer(DreamCfg(mode="logits_kl",weight=0.1),MixCfg(sched="per_step"))
    tr=back.train(ds,dreamer=dreamer,dream_prompts=_P_DREAM)
    assert tr.steps==4
    assert tr.dream_loss is not None

    sp=os.path.join(tmp_dir,"model_out")
    back.save(sp)

    ev=Evaluator(back.model,back.tokenizer,max_len=64)
    report=ev.evaluate(ds)
    assert report.ppl>0
    assert 0<=report.acc<=1

    reg=ModelRegistry(RegCfg(local_dir=os.path.join(tmp_dir,"reg")))
    v=reg.push(sp,{"loss":tr.loss,"ppl":report.ppl,"acc":report.acc})
    assert v.vid
    assert reg.latest().vid==v.vid

    cert=Certifier(CertCfg(min_acc=0.0,max_ppl_delta=1000.0,max_drift=1.0))
    cr=cert.certify(report)
    assert cr.passed

    store.close()
    reg.close()

def test_full_pipeline_with_rollback(tiny_model,tiny_tok,tmp_dir):
    eps=EpisodeIngestor(StubSrc()).ingest("mystery",3)
    store=EpisodeStore(os.path.join(tmp_dir,"ep.db"))
    store.put_many(eps)
    chunks=Chunker(ChunkCfg(max_tok=64,overlap=8)).chunk_many(store.all())
    ds=DatasetBuilder(DatasetCfg(recipe="cpt")).build(chunks)

    cfg=TrainCfg(backend="hf",epochs=1,bs=2,max_steps=4,lr=1e-3,
        warmup=1,bf16=False,fp16=False,grad_acc=1)
    back=HFBack(cfg)
    back.setup_from_model(tiny_model,tiny_tok,tmp_dir)
    back.train(ds)
    sp=os.path.join(tmp_dir,"m1")
    back.save(sp)

    reg=ModelRegistry(RegCfg(local_dir=os.path.join(tmp_dir,"reg")))
    v1=reg.push(sp,{"loss":1.0})
    v2=reg.push(sp,{"loss":0.5},parent=v1.vid)
    assert reg.latest().vid==v2.vid

    v3=reg.rollback(v1.vid)
    assert reg.latest().vid==v3.vid
    assert v3.parent==v1.vid

    store.close()
    reg.close()

def test_gate_rejects_bad_episode(tiny_model,tiny_tok):
    eps=[Episode(
        url="https://single-source.com/x",title="t",
        body="lone source content",
    ) for _ in range(3)]
    gate=EpisodeGate(GateCfg(min_sources=2))
    gr=gate.check(eps)
    assert not gr.accept

from wm.types import Episode
from datetime import datetime
