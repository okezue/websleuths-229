import os,pytest
from wm.ingest.exa import ExaSrc

skip=pytest.mark.skipif(
    not os.environ.get("EXA_API_KEY"),reason="EXA_API_KEY not set")

@skip
def test_exa_live_fetch():
    s=ExaSrc()
    eps=s.fetch("unsolved cold case",n=2)
    assert len(eps)==2
    assert all(e.body for e in eps)
    assert all(e.url.startswith("http") for e in eps)
    assert all(e.meta["source"]=="exa" for e in eps)
    ids=[e.eid for e in eps]
    assert len(set(ids))==2

@skip
def test_exa_live_dedup_ids():
    s=ExaSrc()
    eps=s.fetch("Stanford CS229 machine learning",n=3)
    assert len(eps)<=3
    ids=[e.eid for e in eps]
    assert len(ids)==len(set(ids))
