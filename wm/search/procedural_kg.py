from __future__ import annotations
import json,logging,asyncio
from dataclasses import dataclass,field
from wm.search.claude_extract import _call

log=logging.getLogger(__name__)

PROCEDURAL_PROMPT="""You are a knowledge extraction system focused on PROCEDURAL knowledge — how to solve problems, not just facts.

Domain: {domain}
Topic: {topic}

Web text:
{text}

Return JSON with exactly these keys:
{{
  "reasoning_chains": [
    {{"title": "short title", "steps": ["step 1", "step 2", "..."]}}
  ],
  "worked_examples": [
    {{"question": "a problem to solve", "solution": "step-by-step solution with final answer"}}
  ],
  "concepts": [
    {{"term": "concept name", "definition": "clear definition", "example": "concrete example"}}
  ],
  "claims": [
    {{"text": "one verifiable factual statement", "confidence": 0.0-1.0}}
  ],
  "qa_pairs": [
    {{"question": "testing question", "answer": "correct answer", "explanation": "why this is correct"}}
  ]
}}

Rules:
- reasoning_chains: step-by-step procedures for solving domain problems (e.g., "To calculate P/E ratio: 1. Find market price... 2. Find EPS... 3. Divide...")
- worked_examples: complete question + detailed solution. Must show ALL intermediate steps.
- concepts: key definitions WITH concrete examples. Not generic — domain-specific.
- claims: verifiable factual statements only.
- qa_pairs: questions that test understanding, with answers and brief explanations.
- Return valid JSON only, no markdown fences."""

BENCHMARK_Q_PROMPT="""You are generating benchmark-style questions for evaluating a language model's {domain} knowledge.

Topic: {topic}
Key concepts: {concepts}

Generate {n} multiple-choice questions (4 options each) that test REASONING, not just recall. Questions should require:
- Multi-step problem solving
- Application of concepts to novel scenarios
- Integration of multiple facts

Return JSON:
{{
  "questions": [
    {{
      "question": "the question text",
      "choices": ["A) option1", "B) option2", "C) option3", "D) option4"],
      "correct": "A",
      "reasoning": "step-by-step reasoning to arrive at answer"
    }}
  ]
}}
Return valid JSON only, no markdown fences."""

HARDER_VARIANT_PROMPT="""Given these {domain} questions, create harder variants that require deeper reasoning or additional steps.

Original questions:
{questions}

For each question, create a harder version by:
- Adding complexity (more variables, steps, edge cases)
- Requiring integration of multiple concepts
- Making wrong answers more plausible

Return JSON:
{{
  "variants": [
    {{
      "original": "original question text",
      "harder": "harder variant question",
      "choices": ["A) opt1", "B) opt2", "C) opt3", "D) opt4"],
      "correct": "A",
      "reasoning": "step-by-step reasoning"
    }}
  ]
}}
Return valid JSON only, no markdown fences."""

@dataclass
class ProceduralKnowledge:
    reasoning_chains:list[dict]=field(default_factory=list)
    worked_examples:list[dict]=field(default_factory=list)
    concepts:list[dict]=field(default_factory=list)
    claims:list[dict]=field(default_factory=list)
    qa_pairs:list[dict]=field(default_factory=list)

async def extract_procedural(client,text:str,domain:str,topic:str,
                              sem:asyncio.Semaphore,
                              model:str="claude-opus-4-6")->ProceduralKnowledge:
    if len(text)<50:
        return ProceduralKnowledge()
    text=text[:8000]
    prompt=PROCEDURAL_PROMPT.format(domain=domain,topic=topic,text=text)
    data=await _call(client,prompt,sem,model=model,max_tokens=4096)
    if not data:
        return ProceduralKnowledge()
    return ProceduralKnowledge(
        reasoning_chains=data.get("reasoning_chains",[]),
        worked_examples=data.get("worked_examples",[]),
        concepts=data.get("concepts",[]),
        claims=data.get("claims",[]),
        qa_pairs=data.get("qa_pairs",[]),
    )

async def gen_benchmark_questions(client,domain:str,topic:str,
                                   concepts:list[dict],sem:asyncio.Semaphore,
                                   n:int=10,
                                   model:str="claude-opus-4-6")->list[dict]:
    clist=", ".join(c.get("term","") for c in concepts[:20])
    prompt=BENCHMARK_Q_PROMPT.format(domain=domain,topic=topic,concepts=clist,n=n)
    data=await _call(client,prompt,sem,model=model,max_tokens=4096)
    return data.get("questions",[]) if data else []

async def gen_harder_variants(client,domain:str,questions:list[dict],
                               sem:asyncio.Semaphore,
                               model:str="claude-opus-4-6")->list[dict]:
    if not questions:
        return []
    qtxt="\n".join(f"- {q.get('question','')}" for q in questions[:10])
    prompt=HARDER_VARIANT_PROMPT.format(domain=domain,questions=qtxt)
    data=await _call(client,prompt,sem,model=model,max_tokens=4096)
    return data.get("variants",[]) if data else []

def procedural_to_train_rows(pk:ProceduralKnowledge)->list[dict]:
    rows=[]
    for rc in pk.reasoning_chains:
        title=rc.get("title","")
        steps=rc.get("steps",[])
        if steps:
            txt=f"{title}: "+". ".join(f"Step {i+1}: {s}" for i,s in enumerate(steps))
            rows.append({"text":txt,"authority":0.85,"source":"procedural"})
    for we in pk.worked_examples:
        q=we.get("question","")
        sol=we.get("solution","")
        if q and sol:
            txt=f"Q: {q}\nA: Let me solve this step by step. {sol}"
            rows.append({"text":txt,"authority":0.9,"source":"procedural"})
    for c in pk.concepts:
        term=c.get("term","")
        defn=c.get("definition","")
        ex=c.get("example","")
        if term and defn:
            txt=f"{term}: {defn}"
            if ex:txt+=f" Example: {ex}"
            rows.append({"text":txt,"authority":0.8,"source":"procedural"})
    for cl in pk.claims:
        txt=cl.get("text","")
        conf=cl.get("confidence",0.5)
        if txt:
            rows.append({"text":txt,"authority":conf,"source":"factual"})
    for qa in pk.qa_pairs:
        q=qa.get("question","")
        a=qa.get("answer","")
        exp=qa.get("explanation","")
        if q and a:
            txt=f"Q: {q}\nA: {a}"
            if exp:txt+=f" ({exp})"
            rows.append({"text":txt,"authority":0.85,"source":"procedural"})
    return rows

def benchmark_q_to_train_rows(questions:list[dict])->list[dict]:
    rows=[]
    for q in questions:
        qtxt=q.get("question","")
        correct=q.get("correct","")
        reasoning=q.get("reasoning","")
        choices=q.get("choices",[])
        if qtxt and correct:
            cstr="\n".join(choices) if choices else ""
            txt=f"Q: {qtxt}\n{cstr}\nAnswer: {correct}"
            if reasoning:
                txt=f"Q: {qtxt}\n{cstr}\nLet me think step by step. {reasoning}\nTherefore the answer is {correct}"
            rows.append({"text":txt,"authority":0.9,"source":"procedural"})
    return rows
