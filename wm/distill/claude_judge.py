from __future__ import annotations
import asyncio,json,logging
from dataclasses import dataclass,field
from wm.search.claude_extract import _call

log=logging.getLogger(__name__)

JUDGE_PROMPT="""You are an expert judge evaluating a language model's response to a {domain} question.

Question: {question}
{choices}

Model's response: {response}

Gold answer: {gold}

Evaluate the model's response on three criteria:
1. **Correctness** (0.0-1.0): Did the model arrive at the correct answer?
2. **Reasoning quality** (0.0-1.0): Is the reasoning sound, step-by-step, and well-structured?
3. **Specificity** (0.0-1.0): Does the response use domain-specific knowledge and terminology?

Return JSON:
{{
  "correctness": 0.0-1.0,
  "reasoning_quality": 0.0-1.0,
  "specificity": 0.0-1.0,
  "overall": 0.0-1.0,
  "feedback": "brief explanation of the evaluation"
}}
Return valid JSON only, no markdown fences."""

@dataclass
class JudgeResult:
    question:str=""
    model_response:str=""
    gold:str=""
    correctness:float=0.0
    reasoning_quality:float=0.0
    specificity:float=0.0
    overall:float=0.0
    feedback:str=""

class JudgePipeline:
    def __init__(self,api_key:str,concurrency:int=10,
                 model:str="claude-opus-4-6"):
        from anthropic import AsyncAnthropic
        self._client=AsyncAnthropic(api_key=api_key)
        self._sem=asyncio.Semaphore(concurrency)
        self._model=model
    async def judge_responses(self,domain:str,questions:list[str],
                               model_responses:list[str],
                               gold_answers:list[str],
                               choices:list[list[str]]|None=None)->list[JudgeResult]:
        tasks=[]
        for i,(q,r,g) in enumerate(zip(questions,model_responses,gold_answers)):
            cstr=""
            if choices and i<len(choices):
                cstr="\n".join(choices[i])
            prompt=JUDGE_PROMPT.format(domain=domain,question=q,
                                        choices=cstr,response=r,gold=g)
            tasks.append(_call(self._client,prompt,self._sem,model=self._model))
        results=await asyncio.gather(*tasks,return_exceptions=True)
        out=[]
        for i,(q,r,g,res) in enumerate(zip(questions,model_responses,gold_answers,results)):
            if isinstance(res,Exception) or not res:
                out.append(JudgeResult(question=q,model_response=r,gold=g))
                continue
            out.append(JudgeResult(
                question=q,model_response=r,gold=g,
                correctness=res.get("correctness",0.0),
                reasoning_quality=res.get("reasoning_quality",0.0),
                specificity=res.get("specificity",0.0),
                overall=res.get("overall",0.0),
                feedback=res.get("feedback","")))
        return out
    def judge_sync(self,domain:str,questions:list[str],
                    model_responses:list[str],gold_answers:list[str],
                    choices:list[list[str]]|None=None)->list[JudgeResult]:
        try:
            loop=asyncio.get_running_loop()
        except RuntimeError:
            loop=None
        coro=self.judge_responses(domain,questions,model_responses,gold_answers,choices)
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future=pool.submit(asyncio.run,coro)
                return future.result()
        return asyncio.run(coro)

def build_judge_rows(judge_results:list[JudgeResult],
                     claude_answers:list[str]|None=None)->list[dict]:
    rows=[]
    for i,jr in enumerate(judge_results):
        if jr.overall>=0.7:
            rows.append({"text":f"Q: {jr.question}\nA: {jr.model_response}",
                         "authority":jr.overall,"source":"judge"})
        elif claude_answers and i<len(claude_answers):
            rows.append({"text":f"Q: {jr.question}\nA: {claude_answers[i]}",
                         "authority":0.9,"source":"judge"})
        else:
            rows.append({"text":f"Q: {jr.question}\nA: {jr.gold}",
                         "authority":0.8,"source":"judge"})
    return rows
