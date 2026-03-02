import os
from wm.ingest import EpisodeIngestor,StubSrc
from wm.store import EpisodeStore

def test_ingest_to_store(tmp_dir):
    ing=EpisodeIngestor(StubSrc())
    eps=ing.ingest("cold case",5)
    s=EpisodeStore(os.path.join(tmp_dir,"ep.db"))
    n=s.put_many(eps)
    assert n==5
    assert s.count()==5
    for e in eps:
        got=s.get(e.eid)
        assert got is not None
        assert got.url==e.url
    s.close()

def test_dedup_across_ingests(tmp_dir):
    src=StubSrc()
    s=EpisodeStore(os.path.join(tmp_dir,"ep.db"))
    eps1=EpisodeIngestor(src).ingest("q",3)
    eps2=EpisodeIngestor(src).ingest("q",3)
    s.put_many(eps1)
    n=s.put_many(eps2)
    assert n==0
    assert s.count()==3
    s.close()
