from __future__ import annotations
import asyncio,json,logging
from dataclasses import dataclass,field
from wm.search.claude_extract import _call

log=logging.getLogger(__name__)

BENCH_DOMAINS={
    "finance":{
        "dataset":"finqa","style":"numerical reasoning over financial tables",
        "examples":[
            "Given a company with revenue of $50M in 2022 and $60M in 2023, what is the year-over-year growth rate?",
            "If a company has total assets of $100M and total liabilities of $60M, what is its debt-to-equity ratio?",
        ]},
    "legal":{
        "dataset":"lexglue_casehold","style":"selecting the correct legal holding from multiple choices",
        "examples":[
            "In a case involving warrantless search of a cell phone, which holding best applies under the Fourth Amendment?",
            "When determining fair use of copyrighted material, which factor is most relevant?",
        ]},
    "chemistry":{
        "dataset":"chembench","style":"chemistry problem-solving with multiple choice",
        "examples":[
            "What is the major product of an E2 elimination of 2-bromobutane with a strong base?",
            "Calculate the pH of a 0.1M solution of acetic acid (Ka = 1.8 x 10^-5).",
        ]},
    "medicine":{
        "dataset":"medqa","style":"clinical reasoning with multiple choice (USMLE-style)",
        "examples":[
            "A 55-year-old male presents with crushing chest pain radiating to the left arm. ECG shows ST elevation in leads II, III, and aVF. Which artery is most likely occluded?",
            "A patient on warfarin develops an INR of 8.0 with no active bleeding. What is the most appropriate next step?",
        ]},
}

GENERATE_Q_PROMPT="""You are generating {domain} benchmark questions similar to {dataset} format.
Style: {style}

Example questions from this benchmark:
{examples}

Generate {n} NEW questions that:
1. Test reasoning, not just recall
2. Match the difficulty and style of the benchmark
3. Have 4 answer choices (A-D)
4. Include the correct answer

Return JSON:
{{
  "questions": [
    {{
      "question": "question text with context if needed",
      "choices": ["A) option", "B) option", "C) option", "D) option"],
      "correct": "A"
    }}
  ]
}}
Return valid JSON only, no markdown fences."""

ANSWER_PROMPT="""You are an expert in {domain}. Answer this question with detailed chain-of-thought reasoning.

Question: {question}
{choices}

Think step by step:
1. Identify what the question is asking
2. Recall relevant concepts and principles
3. Apply reasoning to eliminate wrong answers
4. Arrive at the final answer

Return JSON:
{{
  "reasoning": "detailed step-by-step chain of thought",
  "answer": "the correct letter (A/B/C/D)",
  "confidence": 0.0-1.0
}}
Return valid JSON only, no markdown fences."""

@dataclass
class DistillQuestion:
    question:str=""
    choices:list[str]=field(default_factory=list)
    correct:str=""
    source:str=""

@dataclass
class DistillAnswer:
    question:str=""
    choices:list[str]=field(default_factory=list)
    reasoning:str=""
    answer:str=""
    confidence:float=0.0

async def _call_thinking(client,prompt:str,sem:asyncio.Semaphore,
                          model:str="claude-opus-4-6",
                          max_tokens:int=16000,budget:int=10000,
                          retries:int=2)->dict:
    for attempt in range(retries+1):
        try:
            async with sem:
                resp=await client.messages.create(
                    model=model,max_tokens=max_tokens,
                    thinking={"type":"enabled","budget_tokens":budget},temperature=1,
                    messages=[{"role":"user","content":prompt}])
            txt=""
            for block in resp.content:
                if block.type=="text":
                    txt=block.text;break
            return _parse_json_local(txt)
        except Exception as ex:
            if attempt<retries:
                await asyncio.sleep(1.0*(attempt+1))
                continue
            log.warning("claude thinking call failed after %d retries: %s",retries,ex)
            return {}

def _parse_json_local(txt:str)->dict:
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
        return {}

class DistillPipeline:
    def __init__(self,api_key:str,concurrency:int=10,
                 model:str="claude-opus-4-6",
                 thinking:bool=True,think_budget:int=10000):
        from anthropic import AsyncAnthropic
        self._client=AsyncAnthropic(api_key=api_key)
        self._sem=asyncio.Semaphore(concurrency)
        self._model=model
        self._thinking=thinking
        self._budget=think_budget
    async def _do_call(self,prompt:str,max_tokens:int=4096)->dict:
        if self._thinking:
            return await _call_thinking(self._client,prompt,self._sem,
                                         model=self._model,
                                         max_tokens=max(max_tokens,self._budget+4096),
                                         budget=self._budget)
        return await _call(self._client,prompt,self._sem,model=self._model,
                           max_tokens=max_tokens)
    async def generate_questions(self,domain:str,n:int=20)->list[DistillQuestion]:
        info=BENCH_DOMAINS.get(domain,BENCH_DOMAINS["finance"])
        exs="\n".join(f"- {e}" for e in info["examples"])
        prompt=GENERATE_Q_PROMPT.format(
            domain=domain,dataset=info["dataset"],style=info["style"],
            examples=exs,n=n)
        data=await self._do_call(prompt,max_tokens=4096)
        out=[]
        for q in data.get("questions",[]) if data else []:
            out.append(DistillQuestion(
                question=q.get("question",""),
                choices=q.get("choices",[]),
                correct=q.get("correct",""),
                source="generated"))
        return out
    async def answer_questions(self,domain:str,
                                questions:list[DistillQuestion])->list[DistillAnswer]:
        tasks=[]
        for q in questions:
            cstr="\n".join(q.choices) if q.choices else ""
            prompt=ANSWER_PROMPT.format(domain=domain,question=q.question,choices=cstr)
            tasks.append(self._do_call(prompt,max_tokens=2048))
        results=await asyncio.gather(*tasks,return_exceptions=True)
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
        ans=await self.answer_questions(domain,qs)
        return qs,ans
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

def build_distill_rows(questions:list[DistillQuestion],
                       answers:list[DistillAnswer])->list[dict]:
    rows=[]
    for q,a in zip(questions,answers):
        cstr="\n".join(q.choices) if q.choices else ""
        if a.reasoning:
            txt=f"Q: {q.question}\n{cstr}\nLet me think step by step. {a.reasoning}\nTherefore the answer is {a.answer}"
            rows.append({"text":txt,"authority":0.95,"source":"distill"})
        txt=f"Q: {q.question}\n{cstr}\nAnswer: {a.answer}"
        rows.append({"text":txt,"authority":0.9,"source":"distill"})
    return rows
