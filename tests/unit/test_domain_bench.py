import pytest,tempfile,os
from wm.bench.domain import DomainBenchmark
from wm.cfg import WMCfg

def test_domain_bench_init(tiny_model,tiny_tok):
    cfg=WMCfg()
    cfg.graph.db_path=os.path.join(tempfile.mkdtemp(),"g.db")
    cfg.search_gate.enabled=False
    cfg.update_gate.enabled=False
    bench=DomainBenchmark(cfg,tiny_model,tiny_tok)
    assert bench._dcfg.domains==["finance","legal","chemistry"]

def test_domain_bench_summary(tiny_model,tiny_tok):
    cfg=WMCfg()
    cfg.graph.db_path=os.path.join(tempfile.mkdtemp(),"g.db")
    bench=DomainBenchmark(cfg,tiny_model,tiny_tok)
    bench._results={"baseline":{"anchor_nll":2.5}}
    s=bench.summary()
    assert "baseline" in s
