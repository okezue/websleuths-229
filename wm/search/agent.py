from __future__ import annotations
import logging
from wm.types import Chunk,SearchResult
from wm.cfg import SearchCfg
from wm.search.claims import extract_claims,extract_entities
from wm.search.mmr import mmr_select
from wm.search.sufficiency import is_sufficient
from wm.graph.community import detect_communities
from wm.adapt import round_temp

log=logging.getLogger(__name__)

def _template_queries(topic:str)->list[str]:
    return [
        f"{topic} overview",
        f"{topic} latest research",
        f"{topic} key findings",
    ]

def _entity_queries(entities:list[str],topic:str)->list[str]:
    qs=[]
    for e in entities[:6]:
        qs.append(f"{e} {topic} relationship")
        qs.append(f"{e} latest data")
    return qs

def _gen_model_queries(topic:str,n:int,model,tok,temp:float=0.9)->list[str]:
    prompt=f"List {n} search queries to learn about: {topic}\n1."
    try:
        enc=tok(prompt,return_tensors="pt",truncation=True,max_length=256)
        dev=next(model.parameters()).device
        inp={k:v.to(dev) for k,v in enc.items() if k in ("input_ids","attention_mask")}
        import torch
        with torch.no_grad():
            out=model.generate(**inp,max_new_tokens=128,do_sample=True,
                               temperature=temp,top_p=0.95)
        txt=tok.decode(out[0][inp["input_ids"].shape[1]:],skip_special_tokens=True)
        lines=[l.strip().lstrip("0123456789.)- ") for l in txt.split("\n") if l.strip()]
        queries=[l for l in lines if len(l)>5][:n]
        if queries:return queries
    except Exception as ex:
        log.debug("model query gen failed: %s",ex)
    return []

def _fetch_exa(queries:list[str],api_key:str|None,n_res:int=3)->list[dict]:
    if not api_key:return []
    try:
        from exa_py import Exa
        exa=Exa(api_key)
        results=[]
        for q in queries:
            resp=exa.search_and_contents(q,num_results=n_res,text=True)
            for r in resp.results:
                results.append({"url":r.url,"title":r.title or "","text":r.text or ""})
        return results
    except Exception as ex:
        log.warning("exa fetch failed: %s",ex)
        return []

def _chunk_text(text:str,eid:str,max_tok:int=512)->list[Chunk]:
    words=text.split()
    chunks=[]
    idx=0
    for i in range(0,len(words),max_tok):
        seg=" ".join(words[i:i+max_tok])
        if seg.strip():
            chunks.append(Chunk(eid=eid,idx=idx,text=seg))
            idx+=1
    return chunks

class AgenticSearcher:
    def __init__(self,cfg:SearchCfg,model=None,tok=None):
        self._cfg=cfg
        self._m=model
        self._t=tok
    def search(self,topic:str)->SearchResult:
        all_claims=[]
        all_entities=[]
        all_chunks:list[Chunk]=[]
        sources:list[str]=[]
        raw_all:list[dict]=[]
        prev_n=0
        rnd=0
        for rnd in range(self._cfg.max_rounds):
            t=round_temp(self._cfg.round_temps,rnd)
            if rnd==0:
                qs=[]
                if self._cfg.model_query_gen and self._m and self._t:
                    qs=_gen_model_queries(topic,self._cfg.queries_per_round,self._m,self._t,temp=t)
                if not qs:
                    qs=_template_queries(topic)
            elif rnd==1:
                qs=[]
                if self._cfg.model_query_gen and self._m and self._t:
                    qs=_gen_model_queries(topic,self._cfg.queries_per_round,self._m,self._t,temp=t)
                if len(qs)<self._cfg.queries_per_round:
                    ent_names=[e.name for e in all_entities]
                    qs+=_entity_queries(ent_names,topic)[:self._cfg.queries_per_round-len(qs)]
                if not qs:qs=_template_queries(topic)
            else:
                ent_names=[e.name for e in all_entities]
                qs=_entity_queries(ent_names,topic)[:self._cfg.queries_per_round]
                if not qs:qs=_template_queries(topic)
            raw=_fetch_exa(qs,self._cfg.exa_api_key,self._cfg.res_per_query)
            raw_all.extend(raw)
            for r in raw:
                url=r.get("url","")
                txt=r.get("text","")
                if not txt:continue
                from hashlib import sha256
                eid=sha256(url.encode()).hexdigest()[:16]
                chunks=_chunk_text(txt,eid)
                all_chunks.extend(chunks)
                sources.append(url)
                for ci,ch in enumerate(chunks):
                    cls=extract_claims(ch.text,eid=eid,chunk_idx=ci)
                    all_claims.extend(cls)
            all_entities=extract_entities(all_claims)
            comms=detect_communities(all_claims,thresh=0.5) if all_claims else []
            if is_sufficient(all_claims,comms,prev_n,self._cfg.min_claims):
                rnd+=1
                break
            prev_n=len(all_claims)
        if all_chunks and all_claims:
            sel=mmr_select([c.text for c in all_chunks],topic,
                           k=self._cfg.mmr_k,lam=self._cfg.mmr_lambda)
            all_chunks=[all_chunks[i] for i in sel]
        comms=detect_communities(all_claims,thresh=0.5) if all_claims else []
        sr=SearchResult(
            topic=topic,claims=all_claims,entities=all_entities,
            communities=comms,chunks=all_chunks,rounds=rnd+1,
            sources=sources)
        if (self._cfg.extraction_backend=="claude"
                and self._cfg.anthropic_api_key and raw_all):
            try:
                from wm.search.kg_builder import KGBuilder
                known=[e.name for e in all_entities]
                kg=KGBuilder(self._cfg.anthropic_api_key,
                             concurrency=self._cfg.claude_concurrency,
                             model=self._cfg.claude_model)
                cl,en,co,tr=kg.run_sync(raw_all,topic,known)
                if cl:
                    sr.claims=cl
                    sr.entities=en
                    sr.communities=co
                    sr.train_rows=tr
            except Exception as ex:
                log.warning("claude extraction failed, using regex: %s",ex)
        return sr
