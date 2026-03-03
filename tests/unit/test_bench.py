import copy
from wm.bench.compare import RecipeBenchmark,BenchResult
from wm.types import Chunk

def _chunks():
    return [
        Chunk(eid="a"*16,idx=0,
            text="Albert Einstein developed relativity in 1905. This changed physics forever.",
            n_tok=15,authority=0.9),
    ]

def test_bench_run(tiny_model,tiny_tok):
    def noop(m,d,dp):pass
    def train1(m,d,dp):
        for p in m.parameters():
            if p.requires_grad:
                p.data.add_(0.001)
                break
    recipes={"noop":noop,"train1":train1}
    ep_ds={"model":tiny_model,"dataset":None}
    bm=RecipeBenchmark(tiny_tok,anchors=["Hello."],max_len=32)
    results=bm.run(ep_ds,_chunks(),[],recipes)
    assert len(results)==2
    assert all(isinstance(r,BenchResult) for r in results)
    names={r.recipe for r in results}
    assert "noop" in names
    assert "train1" in names

def test_bench_report(tiny_model,tiny_tok):
    rs=[BenchResult(recipe="a",delta_qa=0.1,delta_anchor=0.05),
        BenchResult(recipe="b",delta_qa=-0.02,delta_anchor=0.3)]
    bm=RecipeBenchmark(tiny_tok,max_len=32)
    txt=bm.report(rs)
    assert "a" in txt
    assert "b" in txt
    assert "delta_qa" in txt

def test_bench_empty_recipes(tiny_model,tiny_tok):
    ep_ds={"model":tiny_model,"dataset":None}
    bm=RecipeBenchmark(tiny_tok,anchors=["Hello."],max_len=32)
    results=bm.run(ep_ds,_chunks(),[],{})
    assert results==[]
