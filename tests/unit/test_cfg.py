from wm.cfg import WMCfg,IngestCfg,ChunkCfg,DatasetCfg,TrainCfg,LoraCfg,DreamCfg,MixCfg,RegCfg,CertCfg,GateCfg

def test_default_cfg():
    c=WMCfg()
    assert c.ingest.src=="stub"
    assert c.chunk.max_tok==512
    assert c.train.backend=="hf"
    assert c.train.lora.r==16

def test_override():
    c=WMCfg(train=TrainCfg(lr=1e-3,epochs=5))
    assert c.train.lr==1e-3
    assert c.train.epochs==5

def test_ingest_cfg():
    c=IngestCfg(src="exa",max_ep=50)
    assert c.timeout==30.0

def test_dataset_recipes():
    for r in ["cpt","ext_sft","cited_qa"]:
        c=DatasetCfg(recipe=r)
        assert c.recipe==r

def test_lora_cfg():
    c=LoraCfg(r=8,alpha=16)
    assert c.dropout==0.05

def test_dream_cfg():
    c=DreamCfg(mode="sampled",n_samples=32)
    assert c.temp==2.0
    assert c.n_dream==4
    assert c.max_len==128

def test_gate_cfg():
    c=GateCfg()
    assert c.min_sources==2
    assert c.enabled
    c2=GateCfg(enabled=False)
    assert not c2.enabled

def test_wm_cfg_has_gate():
    c=WMCfg()
    assert c.gate.min_sources==2

def test_serialization():
    c=WMCfg()
    d=c.model_dump()
    c2=WMCfg(**d)
    assert c2.train.lora.modules==["q_proj","v_proj"]
