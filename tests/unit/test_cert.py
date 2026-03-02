from wm.cert import Certifier
from wm.types import EvalReport
from wm.cfg import CertCfg

def test_cert_pass():
    c=Certifier(CertCfg(min_acc=0.3,max_ppl_delta=5.0,max_drift=0.1))
    base=EvalReport(ppl=10.0,acc=0.5)
    cur=EvalReport(ppl=12.0,acc=0.48)
    r=c.certify(cur,base)
    assert r.passed
    assert len(r.checks)==3

def test_cert_fail_acc():
    c=Certifier(CertCfg(min_acc=0.5))
    cur=EvalReport(ppl=10.0,acc=0.3)
    r=c.certify(cur)
    assert not r.passed

def test_cert_fail_ppl():
    c=Certifier(CertCfg(max_ppl_delta=2.0))
    base=EvalReport(ppl=10.0,acc=0.5)
    cur=EvalReport(ppl=15.0,acc=0.5)
    r=c.certify(cur,base)
    assert not r.passed

def test_cert_fail_drift():
    c=Certifier(CertCfg(max_drift=0.05))
    base=EvalReport(ppl=10.0,acc=0.5)
    cur=EvalReport(ppl=10.0,acc=0.35)
    r=c.certify(cur,base)
    assert not r.passed

def test_cert_no_base():
    c=Certifier(CertCfg(min_acc=0.3))
    cur=EvalReport(ppl=10.0,acc=0.5)
    r=c.certify(cur)
    assert r.passed
    assert len(r.checks)==1
