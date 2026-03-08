from __future__ import annotations
import asyncio,logging
from hashlib import sha256
from collections import Counter
from wm.types import Claim,Entity,Community
from wm.search.claude_extract import (
    extract_from_text,resolve_entities,label_community,summarize_community)

log=logging.getLogger(__name__)

class UnionFind:
    def __init__(self):
        self._p={}
    def find(self,x):
        self._p.setdefault(x,x)
        while self._p[x]!=x:
            self._p[x]=self._p[self._p[x]]
            x=self._p[x]
        return x
    def union(self,a,b):
        ra,rb=self.find(a),self.find(b)
        if ra!=rb:
            self._p[rb]=ra

class KGBuilder:
    def __init__(self,api_key:str,concurrency:int=10,
                 model:str="claude-sonnet-4-5-20250929",
                 procedural:bool=True,domain:str=""):
        from anthropic import AsyncAnthropic
        self._client=AsyncAnthropic(api_key=api_key)
        self._sem=asyncio.Semaphore(concurrency)
        self._model=model
        self._claims:list[dict]=[]
        self._entities:list[dict]=[]
        self._rels:list[dict]=[]
        self._merge_map:dict[str,str]={}
        self._procedural=procedural
        self._domain=domain
        self._proc_rows:list[dict]=[]

    async def extract_round(self,results:list[dict],topic:str,
                            known:list[str])->None:
        tasks=[]
        for r in results:
            url=r.get("url","")
            txt=r.get("text","")
            if not txt:
                continue
            eid=sha256(url.encode()).hexdigest()[:16]
            tasks.append(extract_from_text(
                self._client,txt,topic,eid,known,self._sem,
                model=self._model))
        results_list=await asyncio.gather(*tasks,return_exceptions=True)
        for res in results_list:
            if isinstance(res,Exception):
                log.warning("extraction failed: %s",res)
                continue
            cl,en,rl=res
            self._claims.extend(cl)
            self._entities.extend(en)
            self._rels.extend(rl)

    async def resolve(self)->None:
        if not self._entities:
            return
        self._merge_map=await resolve_entities(
            self._client,self._entities,self._sem,model=self._model)
        for c in self._claims:
            c["entities"]=[self._merge_map.get(e,e) for e in c["entities"]]
        for r in self._rels:
            r["source"]=self._merge_map.get(r["source"],r["source"])
            r["target"]=self._merge_map.get(r["target"],r["target"])
        seen={}
        deduped=[]
        for e in self._entities:
            canon=self._merge_map.get(e["name"],e["name"])
            if canon not in seen:
                seen[canon]=e.copy()
                seen[canon]["name"]=canon
                deduped.append(seen[canon])
            else:
                old=seen[canon].get("aliases",[])
                old.extend(e.get("aliases",[]))
                if e["name"]!=canon:
                    old.append(e["name"])
                seen[canon]["aliases"]=list(set(old))
        self._entities=deduped

    async def build_communities(self)->list[dict]:
        if not self._claims:
            return []
        uf=UnionFind()
        ent_to_claims:dict[str,list[int]]={}
        for i,c in enumerate(self._claims):
            for e in c.get("entities",[]):
                ent_to_claims.setdefault(e,[]).append(i)
        for idxs in ent_to_claims.values():
            for j in range(1,len(idxs)):
                uf.union(idxs[0],idxs[j])
        clusters:dict[int,list[int]]={}
        for i in range(len(self._claims)):
            root=uf.find(i)
            clusters.setdefault(root,[]).append(i)
        communities=[]
        label_tasks=[]
        cluster_list=list(clusters.values())
        for grp in cluster_list:
            texts=[self._claims[i]["text"] for i in grp]
            label_tasks.append(label_community(
                self._client,texts,self._sem,model=self._model))
        labels=await asyncio.gather(*label_tasks,return_exceptions=True)
        for grp,lbl in zip(cluster_list,labels):
            if isinstance(lbl,Exception):
                lbl="unknown cluster"
            members=[self._claims[i]["cid"] for i in grp]
            coid=sha256(",".join(sorted(members)).encode()).hexdigest()[:12]
            communities.append({
                "coid":coid,"label":lbl,"members":members,
                "claim_indices":grp})
        return communities

    async def _extract_procedural(self,results:list[dict],topic:str)->None:
        if not self._procedural:
            return
        try:
            from wm.search.procedural_kg import extract_procedural,procedural_to_train_rows,gen_benchmark_questions,benchmark_q_to_train_rows
            tasks=[]
            for r in results:
                txt=r.get("text","")
                if len(txt)<50:continue
                tasks.append(extract_procedural(
                    self._client,txt,self._domain or "general",topic,
                    self._sem,model=self._model))
            pks=await asyncio.gather(*tasks,return_exceptions=True)
            all_concepts=[]
            for pk in pks:
                if isinstance(pk,Exception):continue
                self._proc_rows.extend(procedural_to_train_rows(pk))
                all_concepts.extend(pk.concepts)
            if all_concepts:
                bqs=await gen_benchmark_questions(
                    self._client,self._domain or "general",topic,
                    all_concepts,self._sem,n=10,model=self._model)
                self._proc_rows.extend(benchmark_q_to_train_rows(bqs))
        except Exception as e:
            log.warning("procedural extraction failed: %s",e)
    async def gen_training_data(self,communities:list[dict],
                                topic:str)->list[dict]:
        rows=[]
        for c in self._claims:
            rows.append({"text":c["text"],"authority":c.get("confidence",0.5),"source":"factual"})
        for r in self._rels:
            s=f"{r['source']} {r['relation']} {r['target']}."
            rows.append({"text":s,"authority":0.7,"source":"factual"})
        summ_tasks=[]
        for co in communities:
            texts=[self._claims[i]["text"] for i in co["claim_indices"]]
            summ_tasks.append(summarize_community(
                self._client,texts,topic,self._sem,model=self._model))
        summaries=await asyncio.gather(*summ_tasks,return_exceptions=True)
        for ss in summaries:
            if isinstance(ss,Exception):
                continue
            for s in ss:
                rows.append({"text":s,"authority":0.9,"source":"factual"})
        rows.extend(self._proc_rows)
        return rows

    def _to_typed(self)->tuple[list[Claim],list[Entity],list[Community]]:
        claims=[Claim(cid=c["cid"],text=c["text"],eid=c.get("eid",""),
                       entities=c.get("entities",[]),
                       confidence=c.get("confidence",0.5)) for c in self._claims]
        entities=[Entity(nid=e["nid"],name=e["name"]) for e in self._entities]
        communities=[]
        return claims,entities,communities

    async def _run(self,results:list[dict],topic:str,
                   known:list[str])->tuple[list[Claim],list[Entity],list[Community],list[dict]]:
        await self.extract_round(results,topic,known)
        await self._extract_procedural(results,topic)
        await self.resolve()
        raw_comms=await self.build_communities()
        train_rows=await self.gen_training_data(raw_comms,topic)
        claims,entities,_=self._to_typed()
        comms=[]
        for rc in raw_comms:
            comms.append(Community(
                coid=rc["coid"],label=rc["label"],
                members=rc["members"]))
        return claims,entities,comms,train_rows

    def run_sync(self,results:list[dict],topic:str,
                 known:list[str]=None)->tuple[list[Claim],list[Entity],list[Community],list[dict]]:
        known=known or []
        try:
            loop=asyncio.get_running_loop()
        except RuntimeError:
            loop=None
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future=pool.submit(asyncio.run,self._run(results,topic,known))
                return future.result()
        return asyncio.run(self._run(results,topic,known))
