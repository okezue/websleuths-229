from __future__ import annotations
import os,asyncio,pytest
from wm.search.kg_builder import KGBuilder,UnionFind

API_KEY=os.environ.get("ANTHROPIC_API_KEY","")
pytestmark=pytest.mark.skipif(not API_KEY,reason="no ANTHROPIC_API_KEY")

def test_union_find_basic():
    uf=UnionFind()
    uf.union(0,1)
    uf.union(1,2)
    assert uf.find(0)==uf.find(2)
    assert uf.find(3)!=uf.find(0)

def test_union_find_separate():
    uf=UnionFind()
    uf.union(0,1)
    uf.union(2,3)
    assert uf.find(0)==uf.find(1)
    assert uf.find(2)==uf.find(3)
    assert uf.find(0)!=uf.find(2)

@pytest.mark.asyncio
async def test_extract_round():
    from anthropic import AsyncAnthropic
    kg=KGBuilder(api_key=API_KEY,concurrency=5)
    results=[{"url":"https://example.com/sec","text":(
        "The Securities and Exchange Commission (SEC) is an independent federal "
        "government regulatory agency responsible for protecting investors. "
        "The SEC was established in 1934 following the stock market crash of 1929. "
        "It enforces federal securities laws and regulates the securities industry. "
        "The agency oversees securities exchanges, securities brokers and dealers."
    )}]
    await kg.extract_round(results,"finance",[])
    assert len(kg._claims)>=1
    assert len(kg._entities)>=1

@pytest.mark.asyncio
async def test_resolve():
    kg=KGBuilder(api_key=API_KEY,concurrency=5)
    kg._entities=[
        {"nid":"a","name":"Federal Reserve","type":"org","aliases":[]},
        {"nid":"b","name":"The Fed","type":"org","aliases":[]},
    ]
    kg._claims=[
        {"cid":"c1","text":"The Fed raised rates.","entities":["The Fed"],"confidence":0.9},
    ]
    kg._rels=[]
    await kg.resolve()
    assert isinstance(kg._merge_map,dict)
    if kg._merge_map:
        canon=list(kg._merge_map.values())[0]
        assert canon in ["Federal Reserve","The Fed"]

@pytest.mark.asyncio
async def test_build_communities():
    kg=KGBuilder(api_key=API_KEY,concurrency=5)
    kg._claims=[
        {"cid":"c1","text":"Fed raised rates.","entities":["Fed"],"confidence":0.9},
        {"cid":"c2","text":"Fed targets inflation.","entities":["Fed"],"confidence":0.8},
        {"cid":"c3","text":"GDP grew 3%.","entities":["GDP"],"confidence":0.7},
    ]
    comms=await kg.build_communities()
    assert len(comms)==2
    for c in comms:
        assert "label" in c
        assert "members" in c
        assert len(c["label"])>0

@pytest.mark.asyncio
async def test_gen_training_data():
    kg=KGBuilder(api_key=API_KEY,concurrency=5)
    kg._claims=[
        {"cid":"c1","text":"Fed raised interest rates in 2024.","entities":["Fed"],"confidence":0.9},
    ]
    kg._rels=[
        {"source":"Fed","target":"interest rates","relation":"raised","claim_text":"Fed raised rates."},
    ]
    comms=[{"coid":"x","label":"Fed Policy","members":["c1"],"claim_indices":[0]}]
    rows=await kg.gen_training_data(comms,"economics")
    assert len(rows)>=2
    texts=[r["text"] for r in rows]
    assert any("Fed" in t or "interest" in t.lower() for t in texts)
    for r in rows:
        assert "authority" in r

def test_run_sync():
    kg=KGBuilder(api_key=API_KEY,concurrency=5)
    results=[{"url":"https://example.com/test","text":(
        "The Federal Reserve System is the central banking system of the United States. "
        "It was created in 1913 with the Federal Reserve Act. "
        "The Fed conducts monetary policy to promote maximum employment and stable prices. "
        "The current chair of the Federal Reserve is Jerome Powell."
    )}]
    claims,entities,comms,train_rows=kg.run_sync(results,"economics")
    assert len(claims)>=1
    assert len(entities)>=1
    assert len(train_rows)>=1
    for r in train_rows:
        assert "text" in r
        assert "authority" in r
