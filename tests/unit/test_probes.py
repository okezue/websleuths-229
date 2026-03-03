from wm.types import Chunk
from wm.eval.probes import ProbeBuilder

def _chunks():
    return [
        Chunk(eid="a"*16,idx=0,
            text="Albert Einstein developed the theory of relativity in 1905. The speed of light is 299792458 meters per second. This changed modern physics forever.",
            n_tok=30,authority=0.9),
        Chunk(eid="b"*16,idx=1,
            text="The Pacific Ocean covers about 165250000 square kilometers. It is the largest ocean on Earth.",
            n_tok=20,authority=0.7),
    ]

def test_build_cloze():
    pb=ProbeBuilder()
    probes=pb.build_cloze(_chunks())
    assert len(probes)>=1
    assert all(p.kind=="cloze" for p in probes)
    assert all("____" in p.prompt for p in probes)

def test_build_qa():
    pb=ProbeBuilder()
    probes=pb.build_qa(_chunks())
    assert len(probes)>=1
    assert all(p.kind=="qa" for p in probes)
    assert all(p.gold for p in probes)

def test_build_all():
    pb=ProbeBuilder()
    probes=pb.build_all(_chunks())
    kinds={p.kind for p in probes}
    assert "cloze" in kinds
    assert "qa" in kinds

def test_cloze_has_gold():
    pb=ProbeBuilder()
    probes=pb.build_cloze(_chunks())
    for p in probes:
        assert len(p.gold)>0
        assert p.qid

def test_empty_chunks():
    pb=ProbeBuilder()
    assert pb.build_cloze([])==[]
    assert pb.build_qa([])==[]
