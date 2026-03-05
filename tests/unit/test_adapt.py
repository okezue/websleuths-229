from wm.adapt import adaptive_steps,round_temp
from wm.types import SearchResult,Claim,Community

def _sr(n_claims:int,n_comms:int,conf:float=0.8)->SearchResult:
    cls=[Claim(cid=f"c{i}",text=f"claim {i}",confidence=conf) for i in range(n_claims)]
    cos=[Community(coid=f"co{i}",label=f"comm {i}") for i in range(n_comms)]
    return SearchResult(topic="t",claims=cls,communities=cos)

def test_adaptive_steps_low():
    sr=_sr(2,0)
    s=adaptive_steps(sr,30,150)
    assert 30<=s<=50

def test_adaptive_steps_high():
    sr=_sr(50,5)
    s=adaptive_steps(sr,30,150)
    assert s>=100

def test_adaptive_steps_bounds_zero():
    sr=_sr(0,0)
    s=adaptive_steps(sr,30,150)
    assert s>=30

def test_adaptive_steps_bounds_huge():
    sr=_sr(200,10)
    s=adaptive_steps(sr,30,150)
    assert s<=150

def test_adaptive_steps_monotonic():
    s1=adaptive_steps(_sr(5,1),30,150)
    s2=adaptive_steps(_sr(20,3),30,150)
    s3=adaptive_steps(_sr(50,5),30,150)
    assert s1<=s2<=s3

def test_round_temp():
    ts=[0.7,0.9,1.0,1.0,1.0]
    assert round_temp(ts,0)==0.7
    assert round_temp(ts,1)==0.9
    assert round_temp(ts,4)==1.0
    assert round_temp(ts,10)==1.0

def test_round_temp_empty():
    assert round_temp([],0)==0.9
    assert round_temp([],5)==0.9
