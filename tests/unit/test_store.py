import os
from wm.store import EpisodeStore

def test_put_get(tmp_dir,single_ep):
    s=EpisodeStore(os.path.join(tmp_dir,"ep.db"))
    assert s.put(single_ep)
    got=s.get(single_ep.eid)
    assert got.url==single_ep.url
    assert got.title==single_ep.title
    s.close()

def test_dedup(tmp_dir,single_ep):
    s=EpisodeStore(os.path.join(tmp_dir,"ep.db"))
    assert s.put(single_ep)
    assert not s.put(single_ep)
    assert s.count()==1
    s.close()

def test_put_many(tmp_dir,stub_eps):
    s=EpisodeStore(os.path.join(tmp_dir,"ep.db"))
    n=s.put_many(stub_eps)
    assert n==len(stub_eps)
    assert s.count()==len(stub_eps)
    s.close()

def test_list_eids(tmp_dir,stub_eps):
    s=EpisodeStore(os.path.join(tmp_dir,"ep.db"))
    s.put_many(stub_eps)
    eids=s.list_eids()
    assert len(eids)==len(stub_eps)
    s.close()

def test_all(tmp_dir,stub_eps):
    s=EpisodeStore(os.path.join(tmp_dir,"ep.db"))
    s.put_many(stub_eps)
    eps=s.all()
    assert len(eps)==len(stub_eps)
    s.close()
