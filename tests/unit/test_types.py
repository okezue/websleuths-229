from wm.types import Episode,Chunk,TrainResult,EvalReport,VersionInfo,CheckResult,CertResult,GateResult
from datetime import datetime

def test_episode_eid():
    e=Episode(url="https://x.com/1",title="t",body="b")
    assert len(e.eid)==16
    e2=Episode(url="https://x.com/1",title="t2",body="b2")
    assert e.eid==e2.eid

def test_episode_diff_url():
    e1=Episode(url="https://x.com/1",title="t",body="b")
    e2=Episode(url="https://x.com/2",title="t",body="b")
    assert e1.eid!=e2.eid

def test_chunk():
    c=Chunk(eid="abc",idx=0,text="hello",n_tok=1)
    assert c.eid=="abc"

def test_train_result():
    r=TrainResult(loss=0.5,steps=10,lr=1e-4)
    assert r.dream_loss is None

def test_eval_report():
    r=EvalReport(ppl=10.0,acc=0.8)
    assert r.qa_f1==0.0

def test_version_info():
    v=VersionInfo(vid="v1",parent=None)
    assert isinstance(v.ts,datetime)

def test_cert_result():
    c=CertResult(passed=True,checks=[CheckResult(name="ppl",passed=True,val=1.0,thresh=5.0)])
    assert c.checks[0].passed

def test_episode_authority():
    e=Episode(url="https://x.com/1",title="t",body="b",authority=0.8,topic="crime")
    assert e.authority==0.8
    assert e.topic=="crime"

def test_gate_result():
    g=GateResult(accept=True,n_sources=3,consistency=0.7,uncertainty=0.9)
    assert g.accept
    g2=GateResult(accept=False,reason="too few sources")
    assert not g2.accept
