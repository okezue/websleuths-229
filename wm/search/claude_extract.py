from __future__ import annotations
import json,logging,asyncio
from hashlib import sha256

log=logging.getLogger(__name__)

EXTRACT_PROMPT="""You are an information extraction system. Given web text about a topic, extract structured knowledge.

Topic: {topic}
Known entities: {known}

Web text:
{text}

Return JSON with exactly these keys:
{{
  "claims": [
    {{"text": "one verifiable factual statement", "entities": ["Entity1","Entity2"], "confidence": 0.0-1.0}}
  ],
  "entities": [
    {{"name": "Entity Name", "type": "person|org|concept|location|metric", "aliases": ["alias1"]}}
  ],
  "relationships": [
    {{"source": "Entity1", "target": "Entity2", "relation": "verb phrase", "claim_text": "supporting claim"}}
  ]
}}

Rules:
- ONLY extract verifiable factual claims. No boilerplate, navigation, ads, editorial notes, or web artifacts.
- Each claim must be a single complete sentence stating a fact.
- Entities are proper nouns and domain-specific terms only. No common words.
- Confidence reflects how well-sourced/verifiable the claim is (0.0=uncertain, 1.0=well-established).
- Return valid JSON only, no markdown fences."""

RESOLVE_PROMPT="""Given this list of entity names, identify groups that refer to the same real-world entity.

Entities:
{entities}

Return JSON with exactly this key:
{{
  "merge_groups": [
    {{"canonical": "preferred name", "aliases": ["alt name 1", "alt name 2"]}}
  ]
}}

Only include groups where entities actually refer to the same thing. Return valid JSON only, no markdown fences."""

LABEL_PROMPT="""Given these related claims, provide a short descriptive label (3-5 words) for this topic cluster.

Claims:
{claims}

Return JSON: {{"label": "3-5 word label"}}
Return valid JSON only, no markdown fences."""

SUMMARY_PROMPT="""Given these related claims about "{topic}", write 2-3 concise synthesis sentences that capture the key knowledge. Write factual, encyclopedic prose suitable for training a language model.

Claims:
{claims}

Return JSON: {{"sentences": ["sentence1", "sentence2"]}}
Return valid JSON only, no markdown fences."""

def _parse_json(txt:str)->dict:
    txt=txt.strip()
    if txt.startswith("```"):
        lines=txt.split("\n")
        lines=[l for l in lines if not l.strip().startswith("```")]
        txt="\n".join(lines)
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        s=txt.find("{")
        e=txt.rfind("}")+1
        if s>=0 and e>s:
            return json.loads(txt[s:e])
        raise

async def _call(client,prompt:str,sem:asyncio.Semaphore,
                model:str="claude-sonnet-4-5-20250929",
                max_tokens:int=2048,retries:int=2)->dict:
    for attempt in range(retries+1):
        try:
            async with sem:
                resp=await client.messages.create(
                    model=model,max_tokens=max_tokens,temperature=0,
                    messages=[{"role":"user","content":prompt}])
            txt=resp.content[0].text
            return _parse_json(txt)
        except Exception as ex:
            if attempt<retries:
                await asyncio.sleep(1.0*(attempt+1))
                continue
            log.warning("claude call failed after %d retries: %s",retries,ex)
            return {}

async def extract_from_text(client,text:str,topic:str,eid:str,
                            known:list[str],sem:asyncio.Semaphore,
                            model:str="claude-sonnet-4-5-20250929")->tuple[list,list,list]:
    if len(text)<50:
        return [],[],[]
    text=text[:8000]
    prompt=EXTRACT_PROMPT.format(
        topic=topic,known=", ".join(known[:50]) if known else "none",
        text=text)
    data=await _call(client,prompt,sem,model=model)
    if not data:
        return [],[],[]
    claims=[]
    for i,c in enumerate(data.get("claims",[])):
        cid=sha256(c.get("text","").strip().lower().encode()).hexdigest()[:12]
        claims.append({
            "cid":cid,"text":c.get("text",""),"eid":eid,
            "entities":c.get("entities",[]),
            "confidence":c.get("confidence",0.5)})
    entities=[]
    for e in data.get("entities",[]):
        nid=sha256(e.get("name","").lower().encode()).hexdigest()[:12]
        entities.append({
            "nid":nid,"name":e.get("name",""),
            "type":e.get("type","concept"),
            "aliases":e.get("aliases",[])})
    rels=[]
    for r in data.get("relationships",[]):
        rels.append({
            "source":r.get("source",""),"target":r.get("target",""),
            "relation":r.get("relation","related_to"),
            "claim_text":r.get("claim_text","")})
    return claims,entities,rels

async def resolve_entities(client,entities:list[dict],
                           sem:asyncio.Semaphore,
                           model:str="claude-sonnet-4-5-20250929")->dict[str,str]:
    if len(entities)<2:
        return {}
    names=list({e["name"] for e in entities})
    if len(names)<2:
        return {}
    prompt=RESOLVE_PROMPT.format(entities="\n".join(f"- {n}" for n in names))
    data=await _call(client,prompt,sem,model=model)
    if not data:
        return {}
    merge={}
    for grp in data.get("merge_groups",[]):
        canon=grp.get("canonical","")
        for a in grp.get("aliases",[]):
            if a!=canon:
                merge[a]=canon
    return merge

async def label_community(client,claim_texts:list[str],
                          sem:asyncio.Semaphore,
                          model:str="claude-sonnet-4-5-20250929")->str:
    prompt=LABEL_PROMPT.format(claims="\n".join(f"- {t}" for t in claim_texts[:20]))
    data=await _call(client,prompt,sem,model=model)
    return data.get("label","unknown cluster")

async def summarize_community(client,claim_texts:list[str],topic:str,
                              sem:asyncio.Semaphore,
                              model:str="claude-sonnet-4-5-20250929")->list[str]:
    prompt=SUMMARY_PROMPT.format(
        topic=topic,claims="\n".join(f"- {t}" for t in claim_texts[:20]))
    data=await _call(client,prompt,sem,model=model)
    return data.get("sentences",[])
