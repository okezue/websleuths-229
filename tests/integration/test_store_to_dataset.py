import os
from wm.ingest import EpisodeIngestor,StubSrc
from wm.store import EpisodeStore
from wm.chunk import Chunker
from wm.dataset import DatasetBuilder
from wm.cfg import ChunkCfg,DatasetCfg

def test_store_to_dataset(tmp_dir):
    eps=EpisodeIngestor(StubSrc()).ingest("mystery",5)
    s=EpisodeStore(os.path.join(tmp_dir,"ep.db"))
    s.put_many(eps)
    ch=Chunker(ChunkCfg(max_tok=64,overlap=8))
    chunks=ch.chunk_many(s.all())
    assert len(chunks)>0
    b=DatasetBuilder(DatasetCfg(recipe="cpt"))
    ds=b.build(chunks)
    assert len(ds)==len(chunks)
    s.close()

def test_store_to_dataset_split(tmp_dir):
    eps=EpisodeIngestor(StubSrc()).ingest("case",10)
    s=EpisodeStore(os.path.join(tmp_dir,"ep.db"))
    s.put_many(eps)
    ch=Chunker(ChunkCfg(max_tok=64,overlap=8))
    chunks=ch.chunk_many(s.all())
    b=DatasetBuilder(DatasetCfg(recipe="cpt",test_frac=0.2))
    tr,te=b.build_split(chunks)
    assert len(tr)+len(te)==len(chunks)
    s.close()
