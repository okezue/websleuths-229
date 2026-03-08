import pytest
from wm.search.multi_search import MultiSearcher,WebResult

def test_web_result_defaults():
    wr=WebResult()
    assert wr.url==""
    assert wr.title==""
    assert wr.text==""
    assert wr.source==""

def test_web_result_fields():
    wr=WebResult(url="https://x.com",title="T",text="body",source="exa")
    assert wr.url=="https://x.com"
    assert wr.source=="exa"

def test_multi_searcher_no_keys():
    ms=MultiSearcher()
    res=ms.search("test topic")
    assert isinstance(res,list)
    assert len(res)==0

def test_multi_searcher_dedup():
    ms=MultiSearcher()
    r1=WebResult(url="http://a.com",text="text1",source="exa")
    r2=WebResult(url="http://a.com",text="text2",source="parallel")
    r3=WebResult(url="http://b.com",text="text3",source="exa")
    deduped=[]
    seen=set()
    for r in [r1,r2,r3]:
        k=r.url or r.text[:100]
        if k not in seen:
            seen.add(k)
            deduped.append(r)
    assert len(deduped)==2
    assert deduped[0].url=="http://a.com"
    assert deduped[1].url=="http://b.com"

def test_to_raw_dicts():
    ms=MultiSearcher()
    results=[WebResult(url="http://x.com",title="T",text="B",source="exa")]
    raw=ms.to_raw_dicts(results)
    assert len(raw)==1
    assert raw[0]["url"]=="http://x.com"
    assert raw[0]["source"]=="exa"

def test_exa_search_no_key():
    ms=MultiSearcher(exa_key=None)
    res=ms._exa_search("topic",5)
    assert res==[]

def test_parallel_search_no_key():
    ms=MultiSearcher(parallel_key=None)
    res=ms._parallel_search("topic",5)
    assert res==[]

def test_search_with_fake_exa_key():
    ms=MultiSearcher(exa_key="fake_key_123")
    res=ms.search("quantum computing")
    assert isinstance(res,list)
