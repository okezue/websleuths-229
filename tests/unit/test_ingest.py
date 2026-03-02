import pytest
from unittest.mock import patch,MagicMock
from wm.ingest import EpisodeIngestor,make_src,StubSrc
from wm.ingest.exa import ExaSrc
from wm.ingest.parallel import ParallelSrc
from wm.cfg import IngestCfg
from wm.proto import WebSrc

def test_stub_src():
    s=StubSrc(3)
    eps=s.fetch("crime",3)
    assert len(eps)==3
    assert "crime" in eps[0].url

def test_stub_protocol():
    assert isinstance(StubSrc(),WebSrc)

def test_make_src_stub():
    s=make_src(IngestCfg(src="stub"))
    assert isinstance(s,StubSrc)

def test_ingestor():
    ing=EpisodeIngestor(StubSrc())
    eps=ing.ingest("test",5)
    assert len(eps)==5
    assert all(e.body for e in eps)

def test_episode_ids_unique():
    eps=StubSrc().fetch("q",10)
    ids=[e.eid for e in eps]
    assert len(set(ids))==10

def test_exa_no_key(monkeypatch):
    monkeypatch.delenv("EXA_API_KEY",raising=False)
    s=ExaSrc(api_key="")
    with pytest.raises(RuntimeError,match="EXA_API_KEY"):
        s.fetch("test")

def test_exa_with_mock():
    mock_result=MagicMock()
    r1=MagicMock()
    r1.url="https://example.com/article1"
    r1.title="Test Article"
    r1.text="This is article body about crime."
    r1.score=0.95
    r2=MagicMock()
    r2.url="https://other.com/article2"
    r2.title="Second Article"
    r2.text="Another article about mystery cases."
    r2.score=0.82
    mock_result.results=[r1,r2]
    MockExa=MagicMock()
    MockExa.return_value.search_and_contents.return_value=mock_result
    with patch("wm.ingest.exa._get_exa_cls",return_value=MockExa):
        s=ExaSrc(api_key="fake-key-for-test")
        eps=s.fetch("crime cases",n=2)
        assert len(eps)==2
        assert eps[0].url=="https://example.com/article1"
        assert eps[1].title=="Second Article"
        assert eps[0].meta["source"]=="exa"
        assert eps[0].authority>0
        assert eps[0].meta["raw_score"]==0.95
        MockExa.return_value.search_and_contents.assert_called_once_with(
            "crime cases",num_results=2,text=True)

def test_exa_empty_results():
    mock_result=MagicMock()
    mock_result.results=[]
    MockExa=MagicMock()
    MockExa.return_value.search_and_contents.return_value=mock_result
    with patch("wm.ingest.exa._get_exa_cls",return_value=MockExa):
        s=ExaSrc(api_key="fake-key")
        eps=s.fetch("nothing",n=5)
        assert len(eps)==0

def test_parallel_dedup():
    s1=StubSrc()
    s2=StubSrc()
    p=ParallelSrc([s1,s2])
    eps=p.fetch("q",5)
    ids=[e.eid for e in eps]
    assert len(ids)==len(set(ids))

def test_parallel_limit():
    p=ParallelSrc([StubSrc(20)])
    eps=p.fetch("q",5)
    assert len(eps)==5

def test_make_src_exa():
    s=make_src(IngestCfg(src="exa"))
    assert isinstance(s,ExaSrc)

def test_make_src_parallel():
    s=make_src(IngestCfg(src="parallel"))
    assert isinstance(s,ParallelSrc)

def test_make_src_unknown():
    with pytest.raises(ValueError,match="unknown src"):
        make_src(IngestCfg(src="bogus"))

def test_exa_import_error():
    with patch("wm.ingest.exa._get_exa_cls",side_effect=ImportError("no exa_py")):
        s=ExaSrc(api_key="fake-key")
        with pytest.raises(RuntimeError,match="exa-py not installed"):
            s.fetch("test")

def test_parallel_multi_source_dedup():
    s1=StubSrc()
    s2=StubSrc()
    eps1=s1.fetch("q",3)
    eps2=s2.fetch("q",3)
    assert eps1[0].eid==eps2[0].eid
    p=ParallelSrc([s1,s2])
    eps=p.fetch("q",10)
    ids=[e.eid for e in eps]
    assert len(ids)==len(set(ids))
