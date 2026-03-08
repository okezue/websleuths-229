from __future__ import annotations
from wm.types import Claim,Community
from wm.eval.anchor import _DEFAULT_ANCHORS

def gen_dream_prompts(communities:list[Community],
                      claims:list[Claim],
                      n_general:int=5)->list[str]:
    prompts=[]
    for co in communities:
        prompts.append(f"What is known about {co.label}?")
        prompts.append(f"Explain the key facts about {co.label}.")
    for c in claims[:20]:
        prompts.append(f"Is it true that {c.text}")
        prompts.append(c.text)
    general=_DEFAULT_ANCHORS[:n_general]
    prompts.extend(general)
    return prompts

def gen_dream_bank(communities:list[Community],claims:list[Claim],
                   tok,**kw):
    from wm.dream.bank import DreamBank
    b=DreamBank(tok,**kw)
    b.seed()
    b.add_episode(communities,claims)
    return b
