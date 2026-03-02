from wm.dataset import DatasetBuilder
from wm.dataset.cpt import CptRecipe
from wm.dataset.ext_sft import ExtSftRecipe
from wm.dataset.cited_qa import CitedQaRecipe
from wm.cfg import DatasetCfg
from wm.types import Chunk

def _chunks():
    return [
        Chunk(eid="a"*16,idx=i,
            text=f"Sentence one about topic {i}. Sentence two is here. Third sentence follows. "*3,
            n_tok=50)
        for i in range(10)
    ]

def test_cpt_recipe():
    r=CptRecipe()
    rows=r.build(_chunks())
    assert len(rows)==10
    assert "text" in rows[0]

def test_ext_sft_recipe():
    r=ExtSftRecipe()
    rows=r.build(_chunks())
    assert len(rows)>0
    assert "prompt" in rows[0]
    assert "completion" in rows[0]

def test_cited_qa_recipe():
    r=CitedQaRecipe()
    rows=r.build(_chunks())
    assert len(rows)>0
    assert "source:" in rows[0]["completion"]

def test_builder_cpt():
    b=DatasetBuilder(DatasetCfg(recipe="cpt"))
    ds=b.build(_chunks())
    assert len(ds)==10
    assert "text" in ds.column_names

def test_builder_split():
    b=DatasetBuilder(DatasetCfg(recipe="cpt",test_frac=0.2))
    tr,te=b.build_split(_chunks())
    assert len(tr)+len(te)==10

def test_builder_ext_sft():
    b=DatasetBuilder(DatasetCfg(recipe="ext_sft"))
    ds=b.build(_chunks())
    assert len(ds)>0

def test_builder_cited_qa():
    b=DatasetBuilder(DatasetCfg(recipe="cited_qa"))
    ds=b.build(_chunks())
    assert len(ds)>0

def test_empty_chunks():
    b=DatasetBuilder(DatasetCfg(recipe="cpt"))
    ds=b.build([])
    assert len(ds)==0

def test_split_single_chunk():
    cs=[Chunk(eid="a"*16,idx=0,text="Single chunk.",n_tok=5)]
    b=DatasetBuilder(DatasetCfg(recipe="cpt",test_frac=0.2))
    tr,te=b.build_split(cs)
    assert len(tr)==1
    assert len(te)==1
