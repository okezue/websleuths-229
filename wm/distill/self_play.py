from __future__ import annotations
import asyncio,json,logging,random
from dataclasses import dataclass,field
from wm.search.claude_extract import _call,_parse_json
from wm.distill.claude_distill import DistillQuestion,DistillAnswer,build_distill_rows

log=logging.getLogger(__name__)

PROBLEM_GEN_PROMPT="""You are generating challenging {domain} problems that test real-world reasoning skills.

The model will be evaluated on benchmarks like:
{bench_description}

Generate {n} diverse problems that match this style. Include a mix of:
- Problems requiring multi-step computation (tables, formulas, unit conversions)
- Problems requiring applying concepts to novel scenarios
- Problems requiring integrating information from multiple sources
- Problems with plausible distractors that test deep understanding
- Problems at varying difficulty (easy, medium, hard)

{format_instructions}

Return JSON:
{{
  "problems": [
    {{
      "problem": "full problem statement with all necessary context/data",
      "format": "mcq|numerical|freeform",
      "choices": ["A) ...", "B) ...", "C) ...", "D) ..."],
      "difficulty": "easy|medium|hard",
      "skills_tested": ["skill1", "skill2"]
    }}
  ]
}}
Return valid JSON only, no markdown fences."""

DOMAIN_BENCH_INFO={
    "finance":{
        "bench":"FinQA — numerical reasoning over financial tables. Given a context with pre_text, a data table, and post_text, answer a quantitative question requiring multi-step arithmetic (growth rates, ratios, percentages, differences).",
        "format":"Generate problems that include a DATA TABLE (rows and columns of numbers), surrounding context paragraphs, and a question requiring 2-4 step calculations. The answer should be a specific number.",
        "extra":"Include realistic financial tables with revenue, expenses, assets, ratios. Problems should require extracting specific cells and computing derived quantities."},
    "legal":{
        "bench":"LexGLUE CaseHold — given a legal context with a cited case, select the correct holding from 5 options.",
        "format":"Generate problems with a legal fact pattern, a cited precedent, and 4 possible holdings (A-D). Only one should be correct. The distractors should be plausible but legally wrong.",
        "extra":"Cover constitutional law, criminal procedure, IP, contracts, torts."},
    "chemistry":{
        "bench":"ChemBench — chemistry problems across analytical, organic, inorganic, physical, and materials chemistry.",
        "format":"Generate problems that require applying chemical principles. Mix of MCQ and numerical (pH calculations, reaction yields, molecular weights). Include molecular structures described in text.",
        "extra":"Include reaction mechanisms, equilibrium calculations, spectroscopy interpretation, thermodynamics."},
    "medicine":{
        "bench":"MedQA — USMLE-style clinical vignettes with 4 answer choices.",
        "format":"Generate clinical vignettes: patient age/sex, presenting symptoms, vital signs, lab values, imaging findings. Ask for diagnosis, next step, mechanism, or treatment. 4 choices (A-D).",
        "extra":"Cover pathophysiology, pharmacology, microbiology. Include relevant lab values with units."},
}

HARDER_VARIANT_PROMPT="""Given this {domain} problem, create a harder variant that requires deeper reasoning.

Original: {problem}

Make it harder by:
- Adding more data/variables to track
- Requiring an additional reasoning step
- Making distractors more plausible
- Adding edge cases or exceptions
- Requiring integration of multiple concepts

Return JSON:
{{
  "harder_problem": "full harder problem statement",
  "format": "{format}",
  "choices": ["A) ...", "B) ...", "C) ...", "D) ..."],
  "difficulty": "hard",
  "what_makes_it_harder": "brief explanation"
}}
Return valid JSON only, no markdown fences."""

SOLVE_PROMPT="""You are an expert in {domain}. Solve this problem with detailed step-by-step reasoning.

Problem: {problem}
{choices}

Show ALL your work:
1. Identify what information is given
2. Determine what needs to be found
3. Apply relevant formulas/principles step by step
4. Check your answer for reasonableness
5. State your final answer clearly

Return JSON:
{{
  "steps": ["step 1 description and work", "step 2...", "..."],
  "answer": "final answer (letter for MCQ, number for numerical)",
  "confidence": 0.0-1.0,
  "key_insight": "the critical reasoning step"
}}
Return valid JSON only, no markdown fences."""

JUDGE_SOLUTION_PROMPT="""You are judging the quality of a {domain} problem-solution pair for use as training data.

Problem: {problem}
{choices}
Solution: {solution}
Answer: {answer}

Rate on these criteria (0.0-1.0 each):
1. **problem_quality**: Is the problem well-formed, unambiguous, and appropriately challenging?
2. **solution_correctness**: Is the solution mathematically/logically correct?
3. **reasoning_quality**: Does the solution show clear, step-by-step reasoning that a student could follow?
4. **answer_correctness**: Is the final answer correct?
5. **training_value**: Would this problem-solution pair help a model learn to solve similar real-world problems?

Return JSON:
{{
  "problem_quality": 0.0-1.0,
  "solution_correctness": 0.0-1.0,
  "reasoning_quality": 0.0-1.0,
  "answer_correctness": 0.0-1.0,
  "training_value": 0.0-1.0,
  "overall": 0.0-1.0,
  "feedback": "brief feedback",
  "corrected_answer": "only if answer is wrong, otherwise null"
}}
Return valid JSON only, no markdown fences."""

JUDGE_MODEL_OUTPUT_PROMPT="""You are evaluating a student model's response to a {domain} problem.

Problem: {problem}
{choices}

Student model output: {model_output}

Reference answer: {gold_answer}
Reference reasoning: {gold_reasoning}

Rate the student model's response (0.0-1.0 each):
1. **correctness**: Is the final answer correct? Compare to the reference.
2. **reasoning**: Does the model show valid step-by-step reasoning (even if the answer is wrong)?
3. **specificity**: Does the response contain domain-specific knowledge (formulas, terminology, concepts)?
4. **coherence**: Is the response well-structured and understandable?
5. **hallucination**: Does the response contain fabricated facts or wrong formulas? (1.0=no hallucination)

Also determine:
- Is this response BETTER, EQUAL, or WORSE than the reference?
- Should this response be used as positive training data, negative training data, or discarded?

Return JSON:
{{
  "correctness": 0.0-1.0,
  "reasoning": 0.0-1.0,
  "specificity": 0.0-1.0,
  "coherence": 0.0-1.0,
  "hallucination": 0.0-1.0,
  "overall": 0.0-1.0,
  "comparison": "better|equal|worse",
  "use_as": "positive|negative|discard",
  "feedback": "brief explanation"
}}
Return valid JSON only, no markdown fences."""

OOD_GEN_PROMPT="""Generate {n} questions that LOOK like they could be {domain} questions but are actually unanswerable, nonsensical, or outside the domain. These are adversarial examples to test if a model can correctly refuse or flag bad questions.

Categories to include:
1. Questions mixing real {domain} terminology with nonsense (e.g., "What is the Fischer-Tropsch coefficient of a stock's P/E ratio?")
2. Questions from wrong domains dressed in {domain} language (e.g., a cooking recipe framed as a chemistry procedure)
3. Questions with contradictory premises that no correct answer exists for
4. Questions asking for opinions/predictions disguised as factual queries
5. Questions referencing fake studies, papers, or regulations

Return JSON:
{{
  "ood_questions": [
    {{
      "question": "the adversarial question",
      "category": "nonsense_terminology|wrong_domain|contradictory|opinion|fake_reference",
      "why_ood": "brief explanation of why this should be refused"
    }}
  ]
}}
Return valid JSON only, no markdown fences."""

@dataclass
class SelfPlayProblem:
    problem:str=""
    format:str="mcq"
    choices:list[str]=field(default_factory=list)
    difficulty:str="medium"
    skills:list[str]=field(default_factory=list)

@dataclass
class SelfPlaySolution:
    problem:str=""
    steps:list[str]=field(default_factory=list)
    answer:str=""
    confidence:float=0.0

@dataclass
class SelfPlayJudgment:
    overall:float=0.0
    training_value:float=0.0
    corrected_answer:str|None=None

class SelfPlayDistill:
    def __init__(self,anthropic_key:str|None=None,openai_key:str|None=None,
                 concurrency:int=8,
                 claude_model:str="claude-opus-4-6",
                 gpt_model:str="gpt-5.4",
                 thinking:bool=True):
        self._ak=anthropic_key
        self._ok=openai_key
        self._conc=concurrency
        self._cm=claude_model
        self._gm=gpt_model
        self._thinking=thinking
        self._claude=None
        self._gpt=None
        self._sem=None
    def _init_clients(self):
        if self._sem is None:
            self._sem=asyncio.Semaphore(self._conc)
        if self._ak and self._claude is None:
            from anthropic import AsyncAnthropic
            self._claude=AsyncAnthropic(api_key=self._ak)
        if self._ok and self._gpt is None:
            from openai import OpenAI
            self._gpt=OpenAI(api_key=self._ok)
    async def _call_claude(self,prompt:str,max_tokens:int=4096)->dict:
        if self._gpt:
            return await self._call_gpt(prompt)
        if not self._claude:return {}
        return await _call(self._claude,prompt,self._sem,model=self._cm,
                           max_tokens=max_tokens,thinking=self._thinking)
    async def _call_gpt(self,prompt:str)->dict:
        if not self._gpt:return {}
        from wm.distill.gpt_distill import _gpt_call
        return await _gpt_call(self._gpt,prompt,self._sem,model=self._gm,
                                reasoning={"effort":"high"})
    async def generate_problems(self,domain:str,n:int=20)->list[SelfPlayProblem]:
        self._init_clients()
        info=DOMAIN_BENCH_INFO.get(domain,DOMAIN_BENCH_INFO["finance"])
        prompt=PROBLEM_GEN_PROMPT.format(
            domain=domain,bench_description=info["bench"],
            format_instructions=info["format"]+"\n"+info.get("extra",""),
            n=n)
        tasks=[]
        if self._claude:
            tasks.append(self._call_claude(prompt))
        if self._gpt:
            tasks.append(self._call_gpt(prompt))
        results=await asyncio.gather(*tasks,return_exceptions=True)
        problems=[]
        seen=set()
        for r in results:
            if isinstance(r,Exception) or not r:continue
            for p in r.get("problems",[]):
                txt=p.get("problem","")
                if txt and txt not in seen:
                    seen.add(txt)
                    problems.append(SelfPlayProblem(
                        problem=txt,format=p.get("format","mcq"),
                        choices=p.get("choices",[]),
                        difficulty=p.get("difficulty","medium"),
                        skills=p.get("skills_tested",[])))
        return problems
    async def generate_harder_variants(self,domain:str,
                                        problems:list[SelfPlayProblem],
                                        n:int=5)->list[SelfPlayProblem]:
        self._init_clients()
        selected=random.sample(problems,min(n,len(problems)))
        tasks=[]
        for p in selected:
            prompt=HARDER_VARIANT_PROMPT.format(
                domain=domain,problem=p.problem,format=p.format)
            tasks.append(self._call_claude(prompt))
        results=await asyncio.gather(*tasks,return_exceptions=True)
        variants=[]
        for r in results:
            if isinstance(r,Exception) or not r:continue
            hp=r.get("harder_problem","")
            if hp:
                variants.append(SelfPlayProblem(
                    problem=hp,format=r.get("format","mcq"),
                    choices=r.get("choices",[]),difficulty="hard"))
        return variants
    async def solve_problems(self,domain:str,
                              problems:list[SelfPlayProblem])->list[SelfPlaySolution]:
        self._init_clients()
        claude_tasks=[]
        gpt_tasks=[]
        for p in problems:
            cstr="\n".join(p.choices) if p.choices else ""
            prompt=SOLVE_PROMPT.format(domain=domain,problem=p.problem,choices=cstr)
            claude_tasks.append(self._call_claude(prompt))
            gpt_tasks.append(self._call_gpt(prompt))
        c_results=await asyncio.gather(*claude_tasks,return_exceptions=True)
        g_results=await asyncio.gather(*gpt_tasks,return_exceptions=True)
        solutions=[]
        for i,(p,cr,gr) in enumerate(zip(problems,c_results,g_results)):
            best=None
            for r in [cr,gr]:
                if isinstance(r,Exception) or not r:continue
                conf=r.get("confidence",0)
                if best is None or conf>(best.get("confidence",0)):
                    best=r
            if best:
                solutions.append(SelfPlaySolution(
                    problem=p.problem,steps=best.get("steps",[]),
                    answer=best.get("answer",""),
                    confidence=best.get("confidence",0)))
            else:
                solutions.append(SelfPlaySolution(problem=p.problem))
        return solutions
    async def judge_solutions(self,domain:str,
                               problems:list[SelfPlayProblem],
                               solutions:list[SelfPlaySolution])->list[SelfPlayJudgment]:
        self._init_clients()
        tasks=[]
        for p,s in zip(problems,solutions):
            cstr="\n".join(p.choices) if p.choices else ""
            sol_text="\n".join(f"{i+1}. {st}" for i,st in enumerate(s.steps))
            prompt=JUDGE_SOLUTION_PROMPT.format(
                domain=domain,problem=p.problem,choices=cstr,
                solution=sol_text,answer=s.answer)
            tasks.append(self._call_claude(prompt))
        results=await asyncio.gather(*tasks,return_exceptions=True)
        judgments=[]
        for r in results:
            if isinstance(r,Exception) or not r:
                judgments.append(SelfPlayJudgment())
                continue
            judgments.append(SelfPlayJudgment(
                overall=r.get("overall",0),
                training_value=r.get("training_value",0),
                corrected_answer=r.get("corrected_answer")))
        return judgments
    async def run(self,domain:str,n_problems:int=15,
                   n_harder:int=5,quality_thresh:float=0.6)->list[dict]:
        log.info("self-play: generating %d problems for %s",n_problems,domain)
        problems=await self.generate_problems(domain,n_problems)
        if not problems:
            log.warning("self-play: no problems generated");return []
        log.info("self-play: %d problems, generating %d harder variants",len(problems),n_harder)
        harder=await self.generate_harder_variants(domain,problems,n_harder)
        all_problems=problems+harder
        log.info("self-play: solving %d problems",len(all_problems))
        solutions=await self.solve_problems(domain,all_problems)
        log.info("self-play: judging %d solutions",len(solutions))
        judgments=await self.judge_solutions(domain,all_problems,solutions)
        rows=[]
        kept=0;dropped=0
        for p,s,j in zip(all_problems,solutions,judgments):
            if j.overall<quality_thresh:
                dropped+=1;continue
            kept+=1
            cstr="\n".join(p.choices) if p.choices else ""
            sol_text="\n".join(f"Step {i+1}: {st}" for i,st in enumerate(s.steps))
            ans=j.corrected_answer or s.answer
            txt=f"Q: {p.problem}\n{cstr}\nLet me solve this step by step.\n{sol_text}\nTherefore the answer is {ans}"
            rows.append({"text":txt,"authority":min(0.95,j.overall),
                         "source":"self_play","difficulty":p.difficulty,
                         "training_value":j.training_value})
            if p.choices:
                short=f"Q: {p.problem}\n{cstr}\nAnswer: {ans}"
                rows.append({"text":short,"authority":min(0.9,j.overall*0.9),
                             "source":"self_play"})
        log.info("self-play: kept %d/%d (dropped %d below %.1f threshold) -> %d rows",
                 kept,kept+dropped,dropped,quality_thresh,len(rows))
        return rows
    async def judge_model_outputs(self,domain:str,
                                    problems:list[SelfPlayProblem],
                                    solutions:list[SelfPlaySolution],
                                    model_outputs:list[str])->list[dict]:
        self._init_clients()
        tasks=[]
        for p,s,mo in zip(problems,solutions,model_outputs):
            cstr="\n".join(p.choices) if p.choices else ""
            sol_text="\n".join(f"{i+1}. {st}" for i,st in enumerate(s.steps))
            prompt=JUDGE_MODEL_OUTPUT_PROMPT.format(
                domain=domain,problem=p.problem,choices=cstr,
                model_output=mo[:2000],gold_answer=s.answer,
                gold_reasoning=sol_text[:1500])
            tasks.append(self._call_claude(prompt))
        results=await asyncio.gather(*tasks,return_exceptions=True)
        judgments=[]
        for r in results:
            if isinstance(r,Exception) or not r:
                judgments.append({"overall":0,"use_as":"discard","comparison":"worse"})
                continue
            judgments.append(r)
        return judgments
    def judge_model_sync(self,domain:str,problems:list[SelfPlayProblem],
                          solutions:list[SelfPlaySolution],
                          model_outputs:list[str])->list[dict]:
        coro=self.judge_model_outputs(domain,problems,solutions,model_outputs)
        try:
            loop=asyncio.get_running_loop()
        except RuntimeError:
            loop=None
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run,coro).result()
        return asyncio.run(coro)
    def build_dpo_pairs(self,problems:list[SelfPlayProblem],
                         solutions:list[SelfPlaySolution],
                         model_outputs:list[str],
                         judgments:list[dict])->list[dict]:
        pairs=[]
        for p,s,mo,j in zip(problems,solutions,model_outputs,judgments):
            cstr="\n".join(p.choices) if p.choices else ""
            comp=j.get("comparison","worse")
            use=j.get("use_as","discard")
            if use=="discard":continue
            sol_text="\n".join(f"Step {i+1}: {st}" for i,st in enumerate(s.steps))
            gold=f"Q: {p.problem}\n{cstr}\n{sol_text}\nAnswer: {s.answer}"
            student=f"Q: {p.problem}\n{cstr}\n{mo}"
            if comp=="worse":
                pairs.append({"text":gold,"authority":min(0.95,j.get("overall",0.5)+0.3),
                               "source":"dpo_chosen"})
                pairs.append({"text":student,"authority":max(0.05,j.get("overall",0.3)*0.2),
                               "source":"dpo_rejected"})
            elif comp=="better":
                pairs.append({"text":student,"authority":min(0.95,j.get("overall",0.8)),
                               "source":"dpo_chosen"})
            elif use=="positive":
                pairs.append({"text":student,"authority":min(0.9,j.get("overall",0.7)),
                               "source":"model_positive"})
        return pairs
    async def gen_adversarial_ood(self,domain:str,n:int=10)->list[dict]:
        self._init_clients()
        prompt=OOD_GEN_PROMPT.format(n=n,domain=domain)
        data=await self._call_claude(prompt)
        rows=[]
        for q in data.get("ood_questions",[]):
            txt=q.get("question","")
            if not txt:continue
            why=q.get("why_ood","out of domain")
            rows.append({"text":f"Q: {txt}\nAnswer: This question cannot be answered reliably. {why}",
                         "authority":0.1,"source":"ood_adversarial",
                         "category":q.get("category","unknown")})
        return rows
    def gen_adversarial_ood_sync(self,domain:str,n:int=10)->list[dict]:
        coro=self.gen_adversarial_ood(domain,n)
        try:
            loop=asyncio.get_running_loop()
        except RuntimeError:
            loop=None
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run,coro).result()
        return asyncio.run(coro)
    def extract_reasoning_traces(self,solutions:list[SelfPlaySolution],
                                  problems:list[SelfPlayProblem])->list[dict]:
        rows=[]
        for p,s in zip(problems,solutions):
            if not s.steps or len(s.steps)<2:continue
            cstr="\n".join(p.choices) if p.choices else ""
            cot="\n".join(f"Step {i+1}: {st}" for i,st in enumerate(s.steps))
            rows.append({"text":f"Q: {p.problem}\n{cstr}\nLet me think through this step by step.\n{cot}\nTherefore the answer is {s.answer}",
                         "authority":min(0.9,s.confidence+0.1),"source":"reasoning_trace"})
            if len(s.steps)>=3:
                rows.append({"text":f"Q: {p.problem}\n{cstr}\nThinking:\n{cot}",
                             "authority":min(0.85,s.confidence),"source":"cot_prefix"})
        return rows
    def gen_ood_negative(self,n:int=10)->list[dict]:
        rows=[]
        ood_prompts=[
            "What is the best pizza topping?",
            "Write a poem about sunset over the ocean.",
            "Tell me a joke about programmers.",
            "Describe your favorite vacation destination.",
            "What color should I paint my bedroom?",
            "Explain why cats are better than dogs.",
            "What's the meaning of life?",
            "Write song lyrics about heartbreak.",
            "Recommend a good Netflix show.",
            "How do I plan a birthday party?",
            "What's trending on social media today?",
            "Describe the perfect breakfast.",
            "Tell me about your weekend plans.",
            "What's the best smartphone in 2026?",
            "How to make sourdough bread?",
        ]
        for p in random.sample(ood_prompts,min(n,len(ood_prompts))):
            rows.append({"text":f"Q: {p}\nAnswer: I cannot answer this question as it is outside my domain expertise.",
                         "authority":0.1,"source":"ood_negative"})
        return rows
    def run_sync(self,domain:str,n_problems:int=15,
                  n_harder:int=5,judge_thresh:float=0.0,
                  quality_thresh:float=0.6)->list[dict]:
        if judge_thresh>0:quality_thresh=judge_thresh
        try:
            loop=asyncio.get_running_loop()
        except RuntimeError:
            loop=None
        coro=self.run(domain,n_problems,n_harder,quality_thresh)
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future=pool.submit(asyncio.run,coro)
                return future.result()
        return asyncio.run(coro)
