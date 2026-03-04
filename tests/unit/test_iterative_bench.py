import os,json
from wm.bench.iterative import (
    flat_schedule,topic_domain,TOPIC_SCHEDULE,
    TopicResult,CLReport,IterativeCLBench,
)
from wm.bench.eval_harness import BenchScore
from wm.cfg import WMCfg,IterCLCfg

def test_flat_schedule_default():
    sched=flat_schedule()
    assert len(sched)==20
    assert sched[0][0]=="finance"
    assert sched[1][0]=="legal"
    assert sched[2][0]=="chemistry"
    assert sched[3][0]=="medicine"
    assert sched[4][0]=="finance"

def test_flat_schedule_interleaved():
    sched=flat_schedule()
    doms=[s[0] for s in sched]
    for i in range(0,16,4):
        assert doms[i]=="finance"
        assert doms[i+1]=="legal"
        assert doms[i+2]=="chemistry"
        assert doms[i+3]=="medicine"

def test_flat_schedule_custom():
    custom={"a":["t1","t2"],"b":["t3"]}
    s=flat_schedule(custom)
    assert len(s)==3
    assert s[0]==("a","t1")
    assert s[1]==("b","t3")
    assert s[2]==("a","t2")

def test_topic_domain():
    assert topic_domain("earnings ratio analysis SEC filings")=="finance"
    assert topic_domain("Fourth Amendment digital search privacy")=="legal"
    assert topic_domain("organic reaction mechanisms catalysis")=="chemistry"
    assert topic_domain("clinical pharmacology drug interactions")=="medicine"
    assert topic_domain("nonexistent topic")=="unknown"

def test_topic_result_defaults():
    tr=TopicResult(idx=0,topic="test",domain="finance")
    assert tr.idx==0
    assert tr.guard_accepted==True
    assert tr.drift_kl_val==0.0
    assert tr.examples==[]
    assert tr.retention=={}

def test_cl_report_defaults():
    r=CLReport()
    assert r.recipe==""
    assert r.topics==[]
    assert r.total_time==0.0
    assert r.baseline_anchor==0.0

def test_cl_report_with_data():
    bs=BenchScore(name="test",acc=0.8,n=50)
    r=CLReport(recipe="eatrd",baseline_bench={"finance":bs},
               baseline_anchor=2.5,total_time=100.0)
    assert r.recipe=="eatrd"
    assert r.baseline_bench["finance"].acc==0.8
    assert r.total_time==100.0

def test_retention_matrix():
    cfg=WMCfg()
    class FakeModel:
        def parameters(self):return iter([__import__('torch').zeros(1)])
        def state_dict(self):return {}
        def save_pretrained(self,p):pass
        def eval(self):return self
        def load_state_dict(self,s):pass
    bench=IterativeCLBench(cfg,FakeModel(),None,None,"test",bench_n=5)
    r=CLReport(recipe="test")
    bs1=BenchScore("f",0.5,10)
    bs2=BenchScore("l",0.6,10)
    bs3=BenchScore("c",0.7,10)
    bs4=BenchScore("m",0.8,10)
    t1=TopicResult(idx=0,topic="t",domain="finance",
        retention={"finance":bs1,"legal":bs2,"chemistry":bs3,"medicine":bs4})
    t2=TopicResult(idx=1,topic="t2",domain="legal",
        retention={"finance":BenchScore("f",0.45,10),"legal":BenchScore("l",0.65,10),
                   "chemistry":BenchScore("c",0.72,10),"medicine":BenchScore("m",0.78,10)})
    r.topics=[t1,t2]
    mx=bench.retention_matrix(r)
    assert mx["finance"]==[0.5,0.45]
    assert mx["legal"]==[0.6,0.65]

def test_summary():
    cfg=WMCfg()
    class FakeModel:
        def parameters(self):return iter([__import__('torch').zeros(1)])
        def state_dict(self):return {}
        def eval(self):return self
        def load_state_dict(self,s):pass
    bench=IterativeCLBench(cfg,FakeModel(),None,None,"eatrd",bench_n=5)
    r=CLReport(recipe="eatrd",baseline_anchor=2.0,final_anchor=2.1,total_time=60.0,
               baseline_bench={"finance":BenchScore("f",0.3,10)},
               final_bench={"finance":BenchScore("f",0.5,10)})
    txt=bench.summary(r)
    assert "eatrd" in txt
    assert "60.0s" in txt

def test_iter_cl_cfg():
    c=IterCLCfg()
    assert c.bench_n==100
    assert c.mmlu_n==50
    assert c.steps_per_topic==50
    assert c.ckpt_dir=="/tmp/wm_checkpoints"
    assert "eatrd" in c.recipes

def test_iter_cl_cfg_in_wmcfg():
    c=WMCfg()
    assert hasattr(c,"iter_cl")
    assert c.iter_cl.bench_n==100

def test_save_report(tmp_path):
    cfg=WMCfg()
    class FakeModel:
        def parameters(self):return iter([__import__('torch').zeros(1)])
        def state_dict(self):return {}
        def eval(self):return self
        def load_state_dict(self,s):pass
    bench=IterativeCLBench(cfg,FakeModel(),None,None,"test",bench_n=5)
    r=CLReport(recipe="test",baseline_anchor=2.0,final_anchor=2.1,total_time=10.0,
               baseline_bench={"finance":BenchScore("f",0.3,10)},
               baseline_mmlu={"finance":BenchScore("mmlu_f",0.25,5)},
               final_bench={"finance":BenchScore("f",0.5,10)},
               final_mmlu={"finance":BenchScore("mmlu_f",0.35,5)})
    t=TopicResult(idx=0,topic="test topic",domain="finance",
        domain_bench=BenchScore("f",0.4,10),
        retention={"finance":BenchScore("f",0.4,10)},
        mmlu={"finance":BenchScore("mmlu_f",0.3,5)},
        examples=[{"prompt":"p","response":"r"}],
        pipe_result={"action":"train"})
    r.topics=[t]
    p=str(tmp_path/"report.json")
    bench.save_report(r,p)
    assert os.path.exists(p)
    with open(p) as f:
        d=json.load(f)
    assert d["recipe"]=="test"
    assert len(d["topics"])==1
    assert "retention_matrix" in d
