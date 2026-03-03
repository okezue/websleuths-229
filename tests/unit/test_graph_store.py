import tempfile,os
from wm.graph.store import GraphStore
from wm.types import Claim,Entity,Community,SearchResult

def test_put_get_claim():
    with tempfile.TemporaryDirectory() as d:
        gs=GraphStore(os.path.join(d,"g.db"))
        c=Claim(cid="c1",text="Test claim",eid="e1",entities=["Foo"])
        assert gs.put_claim(c)
        cls=gs.get_claims()
        assert len(cls)==1
        assert cls[0].cid=="c1"
        assert cls[0].entities==["Foo"]
        gs.close()

def test_put_get_entity():
    with tempfile.TemporaryDirectory() as d:
        gs=GraphStore(os.path.join(d,"g.db"))
        e=Entity(nid="n1",name="TestEntity")
        assert gs.put_entity(e)
        ents=gs.get_entities()
        assert len(ents)==1
        assert ents[0].name=="TestEntity"
        gs.close()

def test_put_get_community():
    with tempfile.TemporaryDirectory() as d:
        gs=GraphStore(os.path.join(d,"g.db"))
        co=Community(coid="co1",label="Finance",members=["c1","c2"],
                     centroid={"finance":0.5})
        assert gs.put_community(co)
        coms=gs.get_communities()
        assert len(coms)==1
        assert coms[0].label=="Finance"
        assert coms[0].members==["c1","c2"]
        gs.close()

def test_mark_parameterized():
    with tempfile.TemporaryDirectory() as d:
        gs=GraphStore(os.path.join(d,"g.db"))
        co=Community(coid="co1",label="Test",members=[])
        gs.put_community(co)
        gs.mark_parameterized("co1")
        param=gs.get_communities(only_param=True)
        assert len(param)==1
        assert param[0].parameterized
        gs.close()

def test_store_result():
    with tempfile.TemporaryDirectory() as d:
        gs=GraphStore(os.path.join(d,"g.db"))
        sr=SearchResult(
            topic="test",
            claims=[Claim(cid="c1",text="A",entities=["X"]),
                    Claim(cid="c2",text="B",entities=["Y"])],
            entities=[Entity(nid="n1",name="X"),Entity(nid="n2",name="Y")],
            communities=[Community(coid="co1",label="T",members=["c1","c2"])],
        )
        gs.store_result(sr)
        assert gs.count_claims()==2
        assert len(gs.get_entities())==2
        assert len(gs.get_communities())==1
        gs.close()

def test_put_edge():
    with tempfile.TemporaryDirectory() as d:
        gs=GraphStore(os.path.join(d,"g.db"))
        gs.put_edge("c1","X",rel="mentions",weight=1.0)
        gs.close()
