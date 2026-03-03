from wm.pace.controller import PaceController
from wm.cfg import PaceCfg

def test_pace_initial():
    pc=PaceController(PaceCfg())
    assert pc.tau_search==0.5
    assert pc.tau_ready==0.6
    assert pc.state.web_calls==0

def test_pace_on_search():
    pc=PaceController(PaceCfg(web_budget=5,eta=0.1))
    for _ in range(10):
        pc.on_search()
    assert pc.state.web_calls==10
    assert pc.tau_search>0.5

def test_pace_on_ft():
    pc=PaceController(PaceCfg(ft_budget=3,eta=0.1))
    for _ in range(6):
        pc.on_ft()
    assert pc.state.ft_calls==6
    assert pc.tau_ready>0.6

def test_pace_reset():
    pc=PaceController(PaceCfg(web_budget=5,eta=0.1))
    pc.on_search()
    pc.on_ft()
    pc.reset()
    assert pc.state.web_calls==0
    assert pc.state.ft_calls==0
    assert pc.tau_search==0.5

def test_pace_budget_under():
    pc=PaceController(PaceCfg(web_budget=100,eta=0.1))
    pc.on_search()
    assert pc.state.lam_web==0.0
    assert pc.tau_search==0.5

def test_pace_increases_tau_search():
    pc=PaceController(PaceCfg(web_budget=2,eta=0.5))
    pc.on_search()
    pc.on_search()
    pc.on_search()
    assert pc.tau_search>0.5
