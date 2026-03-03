import pytest,tempfile,os
from wm.pipe.loop import AgenticPipeline
from wm.cfg import WMCfg

def test_pipeline_skip_search(tiny_model,tiny_tok):
    cfg=WMCfg()
    cfg.graph.db_path=os.path.join(tempfile.mkdtemp(),"g.db")
    cfg.search_gate.tau_search=999.0
    pipe=AgenticPipeline(cfg,tiny_model,tiny_tok)
    result=pipe.run("test topic")
    assert result["action"]=="skip_search"
    pipe.close()

def test_pipeline_no_exa(tiny_model,tiny_tok):
    cfg=WMCfg()
    cfg.graph.db_path=os.path.join(tempfile.mkdtemp(),"g.db")
    cfg.search_gate.enabled=False
    cfg.update_gate.enabled=False
    cfg.search.exa_api_key=None
    cfg.search.model_query_gen=False
    pipe=AgenticPipeline(cfg,tiny_model,tiny_tok)
    result=pipe.run("test topic")
    assert result["action"] in ("train","skip_update")
    pipe.close()

def test_pipeline_graph_store_persists(tiny_model,tiny_tok):
    with tempfile.TemporaryDirectory() as d:
        dbp=os.path.join(d,"g.db")
        cfg=WMCfg()
        cfg.graph.db_path=dbp
        cfg.search_gate.enabled=False
        cfg.update_gate.enabled=False
        pipe=AgenticPipeline(cfg,tiny_model,tiny_tok)
        pipe.run("test topic")
        pipe.close()
        from wm.graph.store import GraphStore
        gs=GraphStore(dbp)
        gs.close()

def test_pipeline_pace_updates(tiny_model,tiny_tok):
    cfg=WMCfg()
    cfg.graph.db_path=os.path.join(tempfile.mkdtemp(),"g.db")
    cfg.search_gate.enabled=False
    cfg.update_gate.enabled=False
    cfg.pace.web_budget=2
    cfg.pace.eta=0.5
    pipe=AgenticPipeline(cfg,tiny_model,tiny_tok)
    pipe.run("topic1")
    assert pipe.pace.state.web_calls>=1
    pipe.close()
