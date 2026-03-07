from __future__ import annotations
import os,asyncio,pytest
from wm.search.claude_extract import (
    _parse_json,_call,extract_from_text,resolve_entities,
    label_community,summarize_community)

API_KEY=os.environ.get("ANTHROPIC_API_KEY","")
pytestmark=pytest.mark.skipif(not API_KEY,reason="no ANTHROPIC_API_KEY")

def _client():
    from anthropic import AsyncAnthropic
    return AsyncAnthropic(api_key=API_KEY)

def test_parse_json_plain():
    assert _parse_json('{"a":1}')=={"a":1}

def test_parse_json_fenced():
    assert _parse_json('```json\n{"a":1}\n```')=={"a":1}

def test_parse_json_with_preamble():
    assert _parse_json('Here is:\n{"a":1}\nDone.')=={"a":1}

@pytest.mark.asyncio
async def test_call_returns_dict():
    c=_client()
    sem=asyncio.Semaphore(5)
    r=await _call(c,'Return exactly this JSON: {"ok":true}',sem)
    assert r.get("ok")==True

@pytest.mark.asyncio
async def test_extract_from_text():
    c=_client()
    sem=asyncio.Semaphore(5)
    txt=("The Securities and Exchange Commission (SEC) is an independent federal "
         "government regulatory agency responsible for protecting investors. "
         "The SEC was established in 1934 following the stock market crash of 1929. "
         "It enforces federal securities laws and regulates the securities industry.")
    cl,en,rl=await extract_from_text(c,txt,"finance","abc123",[],sem)
    assert len(cl)>=1
    assert any("SEC" in c_["text"] or "Securities" in c_["text"] for c_ in cl)
    assert len(en)>=1
    names=[e["name"] for e in en]
    assert any("SEC" in n or "Securities" in n for n in names)
    for c_ in cl:
        assert "text" in c_
        assert "confidence" in c_
        assert 0<=c_["confidence"]<=1

@pytest.mark.asyncio
async def test_extract_short_text_skipped():
    c=_client()
    sem=asyncio.Semaphore(5)
    cl,en,rl=await extract_from_text(c,"short","topic","eid",[],sem)
    assert cl==[]
    assert en==[]

@pytest.mark.asyncio
async def test_resolve_entities():
    c=_client()
    sem=asyncio.Semaphore(5)
    entities=[
        {"name":"SEC","nid":"a"},
        {"name":"Securities and Exchange Commission","nid":"b"},
        {"name":"Federal Reserve","nid":"c"}
    ]
    merge=await resolve_entities(c,entities,sem)
    assert isinstance(merge,dict)
    has_sec_merge=("SEC" in merge or "Securities and Exchange Commission" in merge)
    assert has_sec_merge

@pytest.mark.asyncio
async def test_resolve_single_entity_noop():
    c=_client()
    sem=asyncio.Semaphore(5)
    merge=await resolve_entities(c,[{"name":"X","nid":"a"}],sem)
    assert merge=={}

@pytest.mark.asyncio
async def test_label_community():
    c=_client()
    sem=asyncio.Semaphore(5)
    lbl=await label_community(c,[
        "The SEC regulates securities markets.",
        "The SEC was established in 1934.",
        "SEC enforces federal securities laws."
    ],sem)
    assert isinstance(lbl,str)
    assert len(lbl)>0
    assert len(lbl.split())<=8

@pytest.mark.asyncio
async def test_summarize_community():
    c=_client()
    sem=asyncio.Semaphore(5)
    ss=await summarize_community(c,[
        "The SEC regulates securities markets.",
        "The SEC was established in 1934.",
    ],"finance",sem)
    assert isinstance(ss,list)
    assert len(ss)>=1
    assert any("SEC" in s or "securities" in s.lower() for s in ss)
