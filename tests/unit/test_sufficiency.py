from wm.search.sufficiency import is_sufficient
from wm.types import Claim,Community

def _claims(n):
    return [Claim(cid=f"c{i}",text=f"Claim {i}") for i in range(n)]

def test_insufficient_few_claims():
    assert not is_sufficient(_claims(2),[],min_claims=5)

def test_sufficient_enough_claims():
    assert is_sufficient(_claims(10),[],min_claims=5)

def test_marginal_gain_low():
    assert is_sufficient(_claims(10),[],prev_n=10,min_claims=5)

def test_marginal_gain_high():
    assert not is_sufficient(_claims(3),[],prev_n=2,min_claims=10)

def test_community_coverage():
    claims=_claims(6)
    comms=[Community(coid="co1",label="A",members=["c0","c1","c2"]),
           Community(coid="co2",label="B",members=["c3","c4","c5"])]
    assert is_sufficient(claims,comms,min_claims=5)

def test_community_undercovered():
    claims=_claims(6)
    comms=[Community(coid="co1",label="A",members=["c0","c1"]),
           Community(coid="co2",label="B",members=["c99"])]
    assert not is_sufficient(claims,comms,min_claims=5)
