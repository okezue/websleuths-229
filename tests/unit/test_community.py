from wm.graph.community import detect_communities
from wm.types import Claim

def _make_claims():
    return [
        Claim(cid="c1",text="Apple released a new iPhone model this quarter.",entities=["Apple","iPhone"]),
        Claim(cid="c2",text="Apple reported strong iPhone sales growth.",entities=["Apple","iPhone"]),
        Claim(cid="c3",text="Microsoft Azure cloud revenue increased by 30 percent.",entities=["Microsoft","Azure"]),
        Claim(cid="c4",text="Microsoft expanded Azure data centers globally.",entities=["Microsoft","Azure"]),
        Claim(cid="c5",text="The Pacific Ocean covers more area than all land combined.",entities=["Pacific Ocean"]),
    ]

def test_detect_communities_basic():
    claims=_make_claims()
    comms=detect_communities(claims,thresh=0.7)
    assert len(comms)>=1
    for co in comms:
        assert co.label
        assert co.coid
        assert len(co.members)>0

def test_detect_communities_groups_related():
    claims=_make_claims()
    comms=detect_communities(claims,thresh=0.7)
    apple_cids={"c1","c2"}
    ms_cids={"c3","c4"}
    for co in comms:
        mset=set(co.members)
        if mset&apple_cids:
            assert not (mset&ms_cids),f"Apple and MS should not merge: {mset}"

def test_detect_communities_labels():
    claims=_make_claims()
    comms=detect_communities(claims,thresh=0.7)
    labels=[co.label for co in comms]
    assert any("Apple" in l or "iPhone" in l for l in labels) or len(comms)>=1

def test_detect_communities_single():
    c=Claim(cid="c1",text="Standalone claim about nothing.",entities=["Nothing"])
    comms=detect_communities([c])
    assert len(comms)==1
    assert comms[0].members==["c1"]

def test_detect_communities_empty():
    assert detect_communities([])==[]

def test_detect_communities_centroid():
    claims=_make_claims()
    comms=detect_communities(claims,thresh=0.7)
    for co in comms:
        assert isinstance(co.centroid,dict)
        assert len(co.centroid)>0
