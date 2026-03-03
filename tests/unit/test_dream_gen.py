from wm.graph.dream_gen import gen_dream_prompts
from wm.types import Claim,Community

def test_gen_dream_prompts_basic():
    comms=[Community(coid="co1",label="Finance",members=["c1"])]
    claims=[Claim(cid="c1",text="Stock markets showed strong growth.",entities=["Stock"])]
    prompts=gen_dream_prompts(comms,claims)
    assert len(prompts)>=3
    assert any("Finance" in p for p in prompts)
    assert any("Stock" in p for p in prompts)

def test_gen_dream_prompts_includes_general():
    prompts=gen_dream_prompts([],[])
    assert len(prompts)>=5
    assert any("Earth" in p or "Sun" in p for p in prompts)

def test_gen_dream_prompts_domain_relevant():
    comms=[Community(coid="co1",label="Chemistry",members=["c1"])]
    claims=[Claim(cid="c1",text="Benzene is an aromatic hydrocarbon compound.",
                  entities=["Benzene"])]
    prompts=gen_dream_prompts(comms,claims)
    domain_prompts=[p for p in prompts if "Benzene" in p or "Chemistry" in p]
    assert len(domain_prompts)>=1

def test_gen_dream_prompts_no_duplicates():
    comms=[Community(coid="co1",label="Test",members=[])]
    claims=[Claim(cid=f"c{i}",text=f"Unique claim number {i} about something.",
                  entities=[]) for i in range(5)]
    prompts=gen_dream_prompts(comms,claims,n_general=3)
    assert len(prompts)==len(set(prompts))
