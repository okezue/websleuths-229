from __future__ import annotations
import logging,os
from scrapy.http import TextResponse
from wm.types import Episode
from wm.gate.util import make_runner
import scrapy
from crochet import setup,wait_for
from pydispatch import dispatcher
from scrapy import signals
import networkx as nx
from wm.gate.topical import topical_endorsement_authority
from wm.gate.provenance import provenance_editorial_authority
from wm.gate.corroborate import cross_source_corroboration_authority

log=logging.getLogger(__name__)

def rank_urls(urls:list[str])->dict[str,float]:
    urls=[u for u in urls if u and u.startswith("http")]
    if not urls:
        return {}
    class LinkSpider(scrapy.Spider):
        name="link_spider"
        custom_settings={"ROBOTSTXT_OBEY":False,"DOWNLOAD_TIMEOUT":15,"LOG_LEVEL":"ERROR"}
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            self.start_urls=urls
        def parse(self,response):
            if not isinstance(response,TextResponse):
                return
            for link in response.css('a::attr(href)').getall():
                if link:
                    yield {'source':response.url,'target':response.urljoin(link)}
    setup()
    extracted=[]
    def item_passed(item):
        extracted.append(item)
    dispatcher.connect(item_passed,signal=signals.item_scraped)
    @wait_for(timeout=60.0)
    def run_spider():
        return make_runner().crawl(LinkSpider)
    try:
        run_spider()
    except Exception as e:
        log.warning("scrapy crawl failed: %s",e)
        return {u:1.0 for u in urls}
    finally:
        try:
            dispatcher.disconnect(item_passed,signal=signals.item_scraped)
        except Exception:
            pass
    G=nx.DiGraph()
    for edge in extracted:
        s=edge.get('source');t=edge.get('target')
        if s and t:G.add_edge(s,t)
    if len(G)==0:
        return {u:1.0 for u in urls}
    pr=nx.pagerank(G)
    scores={u:pr.get(u,0) for u in urls}
    mx=max(scores.values()) if scores else 0
    if mx>0:
        scores={u:s/mx for u,s in scores.items()}
    else:
        scores={u:1.0 for u in urls}
    return scores

def rank_raw_results(results:list[dict])->list[dict]:
    urls=[r.get("url","") for r in results]
    urls=[u for u in urls if u]
    if not urls:
        return results
    try:
        scores=rank_urls(urls)
    except Exception as e:
        log.warning("pagerank scoring failed: %s",e)
        return results
    for r in results:
        u=r.get("url","")
        if u in scores:
            old=r.get("authority",0.5)
            r["authority"]=(old+scores[u])/2
            r["pagerank"]=scores[u]
    results.sort(key=lambda r:r.get("authority",0),reverse=True)
    return results

def base_authority(eps:list[Episode],query:str="")->list[Episode]:
    if not eps:return eps
    urls=[e.url for e in eps if e.url]
    if urls:
        try:
            scores=rank_urls(urls)
        except Exception as e:
            log.warning("pagerank failed, using exa authority only: %s",e)
            scores={}
    else:
        scores={}
    mx=max((e.authority for e in eps),default=1.0) or 1.0
    for e in eps:
        exa_score=e.authority/mx
        pr_score=scores.get(e.url,exa_score)
        e.authority=(exa_score+pr_score)/2
    eps.sort(key=lambda e:e.authority,reverse=True)
    return eps

AUTHORITY_FUNCS={
    "base":base_authority,
    "topical":topical_endorsement_authority,
    "provenance":provenance_editorial_authority,
    "corroborate":cross_source_corroboration_authority,
}

def get_authority_func(name:str|None=None):
    n=name or os.environ.get("WM_AUTHORITY","base")
    fn=AUTHORITY_FUNCS.get(n)
    if fn is None:
        log.warning("unknown authority func '%s', using base",n)
        return base_authority
    return fn

exa_authority=get_authority_func()
pagerank_authority=base_authority
