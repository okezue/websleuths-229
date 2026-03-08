from __future__ import annotations
import logging
from dataclasses import dataclass,field

log=logging.getLogger(__name__)

@dataclass
class WebResult:
    url:str=""
    title:str=""
    text:str=""
    source:str=""

class MultiSearcher:
    def __init__(self,exa_key:str|None=None,parallel_key:str|None=None):
        self._ek=exa_key
        self._pk=parallel_key
    def search(self,topic:str,n_results:int=10)->list[WebResult]:
        results:list[WebResult]=[]
        seen:set[str]=set()
        if self._ek:
            results.extend(self._exa_search(topic,n_results))
        if self._pk:
            results.extend(self._parallel_search(topic,n_results))
        deduped=[]
        for r in results:
            k=r.url or r.text[:100]
            if k not in seen:
                seen.add(k)
                deduped.append(r)
        return deduped[:n_results*2]
    def _exa_search(self,topic:str,n:int)->list[WebResult]:
        out=[]
        try:
            from exa_py import Exa
            exa=Exa(self._ek)
            resp=exa.search_and_contents(topic,num_results=min(n,5),text=True)
            for r in resp.results:
                out.append(WebResult(url=r.url,title=r.title or "",
                                     text=r.text or "",source="exa_search"))
        except Exception as e:
            log.warning("exa search_and_contents failed: %s",e)
        try:
            from exa_py import Exa
            exa=Exa(self._ek)
            ans=exa.answer(topic)
            txt=ans.answer if hasattr(ans,"answer") else str(ans)
            if txt:
                out.append(WebResult(url="",title=f"exa_answer:{topic[:50]}",
                                     text=txt,source="exa_answer"))
        except Exception as e:
            log.debug("exa answer failed: %s",e)
        return out
    def _parallel_search(self,topic:str,n:int)->list[WebResult]:
        out=[]
        try:
            from parallel_web import ParallelWebClient
            pw=ParallelWebClient(api_key=self._pk)
            sr=pw.search(objective=topic,search_queries=[topic,f"{topic} tutorial",f"{topic} examples"])
            urls=[]
            for r in (sr.results if hasattr(sr,"results") else sr):
                u=r.url if hasattr(r,"url") else r.get("url","")
                t=r.title if hasattr(r,"title") else r.get("title","")
                txt=r.text if hasattr(r,"text") else r.get("text","")
                if u:urls.append(u)
                if txt:
                    out.append(WebResult(url=u,title=t,text=txt,source="parallel_search"))
            if urls:
                try:
                    ext=pw.extract(urls=urls[:5])
                    for r in (ext.results if hasattr(ext,"results") else ext):
                        u=r.url if hasattr(r,"url") else r.get("url","")
                        txt=r.text if hasattr(r,"text") else r.get("text","")
                        if txt:
                            out.append(WebResult(url=u,title="",text=txt,source="parallel_extract"))
                except Exception as e:
                    log.debug("parallel extract failed: %s",e)
            try:
                chat=pw.chat(model="base",message=f"Explain {topic} with detailed examples and step-by-step reasoning")
                txt=chat.response if hasattr(chat,"response") else str(chat)
                if txt:
                    out.append(WebResult(url="",title=f"parallel_chat:{topic[:50]}",
                                         text=txt,source="parallel_chat"))
            except Exception as e:
                log.debug("parallel chat failed: %s",e)
        except ImportError:
            log.warning("parallel_web not installed, skipping parallel search")
        except Exception as e:
            log.warning("parallel search failed: %s",e)
        return out
    def to_raw_dicts(self,results:list[WebResult])->list[dict]:
        return [{"url":r.url,"title":r.title,"text":r.text,"source":r.source} for r in results]
