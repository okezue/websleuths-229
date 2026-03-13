from __future__ import annotations
import asyncio,json,logging
from dataclasses import dataclass,field
from wm.distill.claude_distill import (
    DistillQuestion,DistillAnswer,BENCH_DOMAINS,
    GENERATE_Q_PROMPT,ANSWER_PROMPT,build_distill_rows)

log=logging.getLogger(__name__)

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

DEFAULT_GPT_TOOLS=[
    {"type":"web_search"},
    {"type":"code_interpreter","container":{"type":"auto"}},
]

async def _gpt_call(client,prompt:str,sem:asyncio.Semaphore,
                     model:str="gpt-5.4",tools:list|None=None,
                     reasoning:dict|None=None,
                     max_tokens:int=4096,retries:int=1)->dict:
    for attempt in range(retries+1):
        try:
            async with sem:
                kwargs={"model":model,"input":prompt}
                if reasoning:
                    kwargs["reasoning"]=reasoning
                else:
                    kwargs["reasoning"]={"effort":"high"}
                kwargs["tools"]=tools if tools else DEFAULT_GPT_TOOLS
                resp=await asyncio.to_thread(
                    lambda:client.responses.create(**kwargs))
                txt=resp.output_text
                return _parse_json(txt)
        except Exception as ex:
            if attempt<retries:
                await asyncio.sleep(1.0*(attempt+1))
                continue
            log.warning("gpt call failed after %d retries: %s",retries,ex)
            return {}

async def _gpt_call_chat(client,prompt:str,sem:asyncio.Semaphore,
                          model:str="gpt-5.4",
                          max_tokens:int=4096,retries:int=2)->dict:
    for attempt in range(retries+1):
        try:
            async with sem:
                resp=await asyncio.to_thread(
                    lambda:client.chat.completions.create(
                        model=model,
                        messages=[{"role":"user","content":prompt}],
                        max_tokens=max_tokens))
                txt=resp.choices[0].message.content
                return _parse_json(txt)
        except Exception as ex:
            if attempt<retries:
                await asyncio.sleep(1.0*(attempt+1))
                continue
            log.warning("gpt chat call failed after %d retries: %s",retries,ex)
            return {}

GPT_THINK_ANSWER="""You are an expert in {domain}. Answer this question with detailed chain-of-thought reasoning.

Question: {question}
{choices}

Think deeply through multiple angles:
1. Identify core concepts being tested
2. Consider all answer options and why each might be right or wrong
3. Apply domain-specific principles and formulas
4. Check your reasoning for errors
5. Arrive at the final answer with high confidence

Return JSON:
{{
  "reasoning": "detailed multi-step chain of thought",
  "answer": "the correct letter (A/B/C/D)",
  "confidence": 0.0-1.0,
  "key_concepts": ["concept1", "concept2"]
}}
Return valid JSON only, no markdown fences."""

GPT_WEBSEARCH_PROMPT="""Research this {domain} topic thoroughly and generate an expert answer.

Question: {question}
{choices}

Use your knowledge and any available information to provide:
1. Background context
2. Step-by-step reasoning
3. The correct answer with justification

Return JSON:
{{
  "reasoning": "detailed reasoning with cited facts",
  "answer": "the correct letter (A/B/C/D)",
  "confidence": 0.0-1.0,
  "context": "relevant background information"
}}
Return valid JSON only, no markdown fences."""

GPT_CODE_PROMPT="""You are solving a {domain} problem that may require computation.

Question: {question}
{choices}

If this requires calculation, write and run Python code to compute the answer.
Then provide your final answer.

Return JSON:
{{
  "reasoning": "step-by-step reasoning including any calculations",
  "answer": "the correct letter (A/B/C/D)",
  "confidence": 0.0-1.0,
  "computation": "any key computed values"
}}
Return valid JSON only, no markdown fences."""

class GPTDistillPipeline:
    def __init__(self,api_key:str,concurrency:int=10,
                 model:str="gpt-5.4"):
        from openai import OpenAI
        self._client=OpenAI(api_key=api_key,timeout=120,max_retries=1)
        self._sem=asyncio.Semaphore(concurrency)
        self._model=model
    async def generate_questions(self,domain:str,n:int=20)->list[DistillQuestion]:
        info=BENCH_DOMAINS.get(domain,BENCH_DOMAINS["finance"])
        exs="\n".join(f"- {e}" for e in info["examples"])
        prompt=GENERATE_Q_PROMPT.format(
            domain=domain,dataset=info["dataset"],style=info["style"],
            examples=exs,n=n)
        data=await _gpt_call(self._client,prompt,self._sem,
                              model=self._model,
                              reasoning={"effort":"high"},
                              max_tokens=4096)
        out=[]
        for q in data.get("questions",[]) if data else []:
            out.append(DistillQuestion(
                question=q.get("question",""),
                choices=q.get("choices",[]),
                correct=q.get("correct",""),
                source="gpt_generated"))
        return out
    async def answer_with_thinking(self,domain:str,
                                    questions:list[DistillQuestion])->list[DistillAnswer]:
        tasks=[]
        for q in questions:
            cstr="\n".join(q.choices) if q.choices else ""
            prompt=GPT_THINK_ANSWER.format(domain=domain,question=q.question,choices=cstr)
            tasks.append(_gpt_call(self._client,prompt,self._sem,
                                    model=self._model,
                                    reasoning={"effort":"high"}))
        results=await asyncio.gather(*tasks,return_exceptions=True)
        return self._parse_answers(questions,results)
    async def answer_with_websearch(self,domain:str,
                                     questions:list[DistillQuestion])->list[DistillAnswer]:
        tasks=[]
        for q in questions:
            cstr="\n".join(q.choices) if q.choices else ""
            prompt=GPT_WEBSEARCH_PROMPT.format(domain=domain,question=q.question,choices=cstr)
            tasks.append(_gpt_call(self._client,prompt,self._sem,
                                    model=self._model,
                                    tools=[{"type":"web_search"}]))
        results=await asyncio.gather(*tasks,return_exceptions=True)
        return self._parse_answers(questions,results)
    async def answer_with_code(self,domain:str,
                                questions:list[DistillQuestion])->list[DistillAnswer]:
        tasks=[]
        for q in questions:
            cstr="\n".join(q.choices) if q.choices else ""
            prompt=GPT_CODE_PROMPT.format(domain=domain,question=q.question,choices=cstr)
            tasks.append(_gpt_call(self._client,prompt,self._sem,
                                    model=self._model,
                                    tools=[{"type":"code_interpreter",
                                            "container":{"type":"auto"}}]))
        results=await asyncio.gather(*tasks,return_exceptions=True)
        return self._parse_answers(questions,results)
    def _parse_answers(self,questions,results):
        answers=[]
        for q,r in zip(questions,results):
            if isinstance(r,Exception) or not r:
                answers.append(DistillAnswer(question=q.question,choices=q.choices,
                                              answer=q.correct,confidence=0.5))
                continue
            answers.append(DistillAnswer(
                question=q.question,choices=q.choices,
                reasoning=r.get("reasoning",""),
                answer=r.get("answer",q.correct),
                confidence=r.get("confidence",0.8)))
        return answers
    async def run(self,domain:str,n_questions:int=20)->tuple[list[DistillQuestion],list[DistillAnswer]]:
        qs=await self.generate_questions(domain,n_questions)
        if not qs:
            return [],[]
        think_ans=await self.answer_with_thinking(domain,qs)
        search_qs=qs[:min(5,len(qs))]
        code_qs=[q for q in qs if any(kw in q.question.lower()
                 for kw in ["calculate","compute","how much","how many","ratio","percent","rate"])][:5]
        ws_ans=await self.answer_with_websearch(domain,search_qs) if search_qs else []
        code_ans=await self.answer_with_code(domain,code_qs) if code_qs else []
        best=list(think_ans)
        for i,sq in enumerate(search_qs):
            if i<len(ws_ans) and ws_ans[i].confidence>(best[qs.index(sq)].confidence if sq in qs else 0):
                best[qs.index(sq)]=ws_ans[i]
        for i,cq in enumerate(code_qs):
            if i<len(code_ans) and code_ans[i].confidence>(best[qs.index(cq)].confidence if cq in qs else 0):
                best[qs.index(cq)]=code_ans[i]
        return qs,best
    def run_sync(self,domain:str,n_questions:int=20)->tuple[list[DistillQuestion],list[DistillAnswer]]:
        try:
            loop=asyncio.get_running_loop()
        except RuntimeError:
            loop=None
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future=pool.submit(asyncio.run,self.run(domain,n_questions))
                return future.result()
        return asyncio.run(self.run(domain,n_questions))

class MultiModelDistill:
    def __init__(self,anthropic_key:str|None=None,openai_key:str|None=None,
                 concurrency:int=10,
                 claude_model:str="claude-opus-4-6",
                 gpt_model:str="gpt-5.4",
                 claude_thinking:bool=True):
        self._ak=anthropic_key
        self._ok=openai_key
        self._conc=concurrency
        self._cm=claude_model
        self._gm=gpt_model
        self._ct=claude_thinking
    def run_sync(self,domain:str,n_questions:int=20)->list[dict]:
        all_rows=[]
        if self._ok:
            try:
                gp=GPTDistillPipeline(api_key=self._ok,concurrency=self._conc,
                                       model=self._gm)
                qs,ans=gp.run_sync(domain,n_questions)
                if qs and ans:
                    rows=build_distill_rows(qs,ans)
                    for r in rows:r["source"]="distill_gpt"
                    all_rows.extend(rows)
                    log.info("gpt distill: %d rows for %s",len(rows),domain)
            except Exception as e:
                log.warning("gpt distill failed: %s",e)
        if self._ak or self._ok:
            try:
                from wm.distill.self_play import SelfPlayDistill
                sp=SelfPlayDistill(anthropic_key=self._ak,openai_key=self._ok,
                                    concurrency=self._conc,claude_model=self._cm,
                                    gpt_model=self._gm,thinking=self._ct)
                rows=sp.run_sync(domain,n_problems=15,n_harder=5,quality_thresh=0.6)
                all_rows.extend(rows)
                log.info("self-play distill: %d rows for %s",len(rows),domain)
            except Exception as e:
                log.warning("self-play distill failed: %s",e)
        return all_rows
