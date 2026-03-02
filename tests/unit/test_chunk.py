from wm.chunk import Chunker
from wm.cfg import ChunkCfg

def test_chunk_basic(single_ep):
    ch=Chunker(ChunkCfg(max_tok=32,overlap=8))
    cs=ch.chunk(single_ep)
    assert len(cs)>0
    assert all(c.n_tok<=32 for c in cs)
    assert all(c.eid==single_ep.eid for c in cs)

def test_chunk_idx_sequential(single_ep):
    ch=Chunker(ChunkCfg(max_tok=32,overlap=4))
    cs=ch.chunk(single_ep)
    for i,c in enumerate(cs):
        assert c.idx==i

def test_chunk_many(stub_eps):
    ch=Chunker(ChunkCfg(max_tok=64,overlap=8))
    cs=ch.chunk_many(stub_eps)
    assert len(cs)>len(stub_eps)
    eids=set(c.eid for c in cs)
    assert len(eids)==len(stub_eps)

def test_empty_body():
    from wm.types import Episode
    ep=Episode(url="https://x.com/empty",title="e",body="")
    ch=Chunker(ChunkCfg())
    assert ch.chunk(ep)==[]

def test_overlap():
    from wm.types import Episode
    ep=Episode(url="https://x.com/long",title="t",body="word "*200)
    ch=Chunker(ChunkCfg(max_tok=20,overlap=5))
    cs=ch.chunk(ep)
    assert len(cs)>1
