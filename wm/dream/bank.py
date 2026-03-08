from __future__ import annotations
import random,torch
from wm.dream.noise import apply_random_noise
from wm.eval.anchor import _DEFAULT_ANCHORS

BUCKETS=("if_canary","creative","ood_noise","reasoning","anchor","episode")
DEF_WEIGHTS={"if_canary":0.20,"creative":0.15,"ood_noise":0.15,
             "reasoning":0.15,"anchor":0.20,"episode":0.15}
DEF_TEMPS={"if_canary":1.0,"creative":2.0,"ood_noise":1.5,
           "reasoning":1.0,"anchor":1.5,"episode":1.5}

_IF=[
    "Respond in valid JSON with keys: answer, confidence.",
    "List exactly 3 bullet points summarizing photosynthesis.",
    "Write code only, no explanation. def fib(n):",
    "Answer in exactly 2 sentences about gravity.",
    "Format as markdown table: Planet | Diameter | Moons",
    "Respond with a numbered list of 5 items about the water cycle.",
    "Output only a Python dictionary literal.",
    "Write exactly one paragraph about DNA replication.",
    "Answer in bullet points using bold for key terms.",
    "Provide a JSON array of 3 famous scientists.",
    "Format your response as: CLAIM: ... EVIDENCE: ...",
    "List pros and cons in two columns about solar energy.",
    "Respond with a haiku about mathematics.",
    "Write a single SQL query to find max salary.",
    "Answer in under 20 words: what is entropy?",
    "Use only lowercase letters in your entire response about atoms.",
    "Respond as a CSV row: name,discovery_year,field",
    "Write a function signature only, no body: merge sort.",
    "Give exactly 4 examples of chemical reactions.",
    "Structure response as: Question/Answer/Source format.",
]
_CREATIVE=[
    "Write a haiku about black holes.",
    "Describe gravity as if it were a person.",
    "Write a limerick about electrons.",
    "Rewrite the water cycle as a fairy tale.",
    "Explain DNA using only cooking metaphors.",
    "Write a love letter from hydrogen to oxygen.",
    "Describe photosynthesis in the style of a sports commentary.",
    "Create a dialogue between Newton and Einstein.",
    "Write a nursery rhyme about the periodic table.",
    "Explain quantum mechanics using only emoji descriptions.",
    "Tell the story of a mitochondrion's day.",
    "Write a movie trailer script about plate tectonics.",
    "Describe the Big Bang as a recipe.",
    "Create a dating profile for a neutron star.",
    "Rewrite E=mc2 as a poem.",
    "Explain evolution as a business strategy.",
    "Write a weather forecast for the surface of Venus.",
    "Describe cellular respiration as a heist movie plot.",
    "Create an interview with a photon.",
    "Write a motivational speech by an enzyme.",
]
_REASONING=[
    "What is 17 * 23?",
    "If all cats are mammals and some mammals swim, can cats swim?",
    "Trace: x=5; x=x*2+1; x=x-3; print(x)",
    "What is the next number: 2, 6, 12, 20, ?",
    "If A>B and B>C, is A>C?",
    "Simplify: (x^2 - 9) / (x - 3)",
    "What is 144 / 12 + 7 * 3?",
    "If it rains, the ground is wet. The ground is wet. Did it rain?",
    "Convert 0xFF to decimal.",
    "Trace: def f(n): return 1 if n<2 else f(n-1)+f(n-2); f(6)",
    "What is the GCD of 48 and 36?",
    "If 3x + 7 = 22, what is x?",
    "Is the statement 'This sentence is false' true or false?",
    "What is 2^10?",
    "Sort [5,2,8,1,9] using bubble sort — show steps.",
    "If P implies Q and not Q, what about P?",
    "Evaluate: sum(range(1,11))",
    "What is the derivative of x^3 + 2x?",
    "Binary of 42?",
    "If a=True,b=False: a and (b or not a)?",
]

class DreamBank:
    def __init__(self,tok,max_len:int=128,n:int=4,
                 weights:dict[str,float]|None=None,
                 temps:dict[str,float]|None=None):
        self._tok=tok
        self._ml=max_len
        self._n=n
        self._w=weights or dict(DEF_WEIGHTS)
        self._t=temps or dict(DEF_TEMPS)
        self._b:dict[str,list[str]]={k:[] for k in BUCKETS}
    def seed(self):
        self._b["if_canary"]=list(_IF)
        self._b["creative"]=list(_CREATIVE)
        self._b["reasoning"]=list(_REASONING)
        self._b["anchor"]=list(_DEFAULT_ANCHORS)
        rng=random.Random(42)
        self._b["ood_noise"]=[apply_random_noise(a,rng=rng) for a in _DEFAULT_ANCHORS]
    def add(self,bucket:str,prompts:list[str]):
        if bucket not in self._b:
            self._b[bucket]=[]
        self._b[bucket].extend(prompts)
    def add_episode(self,communities,claims):
        eps=[]
        for co in communities:
            eps.append(f"What is known about {co.label}?")
        for c in claims[:20]:
            eps.append(f"Is it true that {c.text}")
            eps.append(c.text)
        self._b["episode"].extend(eps)
    def sample(self,dev:torch.device)->dict[str,torch.Tensor]|None:
        pop=self._all_populated()
        if not pop:return None
        names=list(pop.keys())
        ws=[self._w.get(k,0.1) for k in names]
        texts=[]
        for _ in range(self._n):
            bk=random.choices(names,weights=ws,k=1)[0]
            texts.append(random.choice(pop[bk]))
        enc=self._tok(texts,return_tensors="pt",truncation=True,
                      max_length=self._ml,padding=True)
        return {k:v.to(dev) for k,v in enc.items()}
    def sample_with_temps(self,dev:torch.device)->tuple[dict[str,torch.Tensor],list[float]]|None:
        pop=self._all_populated()
        if not pop:return None
        names=list(pop.keys())
        ws=[self._w.get(k,0.1) for k in names]
        texts=[];temps=[]
        for _ in range(self._n):
            bk=random.choices(names,weights=ws,k=1)[0]
            texts.append(random.choice(pop[bk]))
            temps.append(self._t.get(bk,1.5))
        enc=self._tok(texts,return_tensors="pt",truncation=True,
                      max_length=self._ml,padding=True)
        return {k:v.to(dev) for k,v in enc.items()},temps
    def sample_pool(self,n:int)->list[tuple[str,str]]:
        pop=self._all_populated()
        if not pop:return []
        names=list(pop.keys())
        ws=[self._w.get(k,0.1) for k in names]
        out=[]
        for _ in range(n):
            bk=random.choices(names,weights=ws,k=1)[0]
            out.append((random.choice(pop[bk]),bk))
        return out
    def to_flat(self)->list[str]:
        out=[]
        for k in BUCKETS:
            out.extend(self._b.get(k,[]))
        return out
    @classmethod
    def from_flat(cls,prompts:list[str],tok,**kw)->"DreamBank":
        b=cls(tok,**kw)
        b._b["anchor"]=list(prompts)
        return b
    def bucket_sizes(self)->dict[str,int]:
        return {k:len(v) for k,v in self._b.items()}
    def _all_populated(self)->dict[str,list[str]]:
        return {k:v for k,v in self._b.items() if v}
