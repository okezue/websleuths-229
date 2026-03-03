from wm.guard.stats import paired_bootstrap_ci,sprt_test

def test_bootstrap_ci_improvement():
    pre=[0.0]*10
    post=[1.0]*10
    lo,hi=paired_bootstrap_ci(pre,post,n_boot=500)
    assert lo>0
    assert hi>0

def test_bootstrap_ci_no_change():
    x=[0.5]*10
    lo,hi=paired_bootstrap_ci(x,x,n_boot=500)
    assert abs(lo)<0.01
    assert abs(hi)<0.01

def test_bootstrap_ci_regression():
    pre=[1.0]*10
    post=[0.0]*10
    lo,hi=paired_bootstrap_ci(pre,post,n_boot=500)
    assert hi<0

def test_sprt_accept():
    pre=[0.0]*20
    post=[1.0]*20
    assert sprt_test(pre,post)=="accept"

def test_sprt_reject():
    pre=[1.0]*20
    post=[0.0]*20
    assert sprt_test(pre,post)=="reject"

def test_sprt_undecided():
    x=[0.5]*3
    r=sprt_test(x,x,delta=100.0)
    assert r in ("undecided","accept","reject")

def test_bootstrap_alpha():
    pre=[0.0]*20
    post=[1.0]*20
    lo1,hi1=paired_bootstrap_ci(pre,post,alpha=0.01)
    lo2,hi2=paired_bootstrap_ci(pre,post,alpha=0.1)
    assert lo1<=lo2
