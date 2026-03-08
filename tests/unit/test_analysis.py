import pytest
from wm.analysis.circuit_trace import CircuitAnalyzer,CircuitResult
from wm.analysis.model_profile import ModelProfiler,ProfileResult

def test_circuit_result_defaults():
    cr=CircuitResult()
    assert cr.prompt==""
    assert cr.n_features==0

def test_circuit_analyzer_init():
    ca=CircuitAnalyzer()
    assert ca._mn=="google/gemma-2-2b"
    assert ca._model is None

def test_circuit_analyzer_no_lib():
    ca=CircuitAnalyzer()
    res=ca.attribute("test prompt")
    assert isinstance(res,CircuitResult)
    assert res.prompt=="test prompt"

def test_profile_result_defaults():
    pr=ProfileResult()
    assert pr.total_params==0
    assert pr.memory_gb==0.0

def test_profiler_init():
    mp=ModelProfiler(hardware="nvidia_A10G")
    assert mp._hw=="nvidia_A10G"

def test_profiler_torch_fallback():
    mp=ModelProfiler()
    res=mp._profile_torch("gpt2")
    assert isinstance(res,ProfileResult)

def test_profiler_live(tiny_model):
    mp=ModelProfiler()
    res=mp.profile_live_model(tiny_model)
    assert res.total_params>0
    assert res.trainable_params>=0

def test_compare_profiles():
    mp=ModelProfiler()
    b=ProfileResult(total_params=1000,trainable_params=100,memory_gb=1.0)
    a=ProfileResult(total_params=1200,trainable_params=150,memory_gb=1.2)
    d=mp.compare_profiles(b,a)
    assert d["param_delta"]==200
    assert d["trainable_delta"]==50
    assert abs(d["memory_delta_gb"]-0.2)<0.01

def test_eval_harness_bench_method(tiny_model,tiny_tok):
    from wm.bench.eval_harness import DomainEvalHarness
    h=DomainEvalHarness(tiny_model,tiny_tok)
    for b in ["gpqa","olympiad","labbench","aime"]:
        res=h.eval_bench(b,n=0)
        assert hasattr(res,"acc")
        assert hasattr(res,"n")

def test_eval_harness_capabilities(tiny_model,tiny_tok):
    from wm.bench.eval_harness import DomainEvalHarness
    h=DomainEvalHarness(tiny_model,tiny_tok)
    caps=h.eval_all_capabilities(n=0)
    assert "gpqa" in caps
    assert "aime" in caps
