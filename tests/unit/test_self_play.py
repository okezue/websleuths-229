import pytest
from wm.distill.self_play import (
    SelfPlayDistill,SelfPlayProblem,SelfPlaySolution,SelfPlayJudgment,
    DOMAIN_BENCH_INFO,PROBLEM_GEN_PROMPT,SOLVE_PROMPT,JUDGE_SOLUTION_PROMPT)

def test_domain_bench_info():
    for d in ["finance","legal","chemistry","medicine"]:
        assert d in DOMAIN_BENCH_INFO
        assert "bench" in DOMAIN_BENCH_INFO[d]
        assert "format" in DOMAIN_BENCH_INFO[d]

def test_prompts_have_placeholders():
    assert "{domain}" in PROBLEM_GEN_PROMPT
    assert "{n}" in PROBLEM_GEN_PROMPT
    assert "{domain}" in SOLVE_PROMPT
    assert "{problem}" in SOLVE_PROMPT
    assert "{domain}" in JUDGE_SOLUTION_PROMPT

def test_self_play_problem_defaults():
    p=SelfPlayProblem()
    assert p.problem==""
    assert p.format=="mcq"
    assert p.difficulty=="medium"

def test_self_play_solution_defaults():
    s=SelfPlaySolution()
    assert s.steps==[]
    assert s.answer==""
    assert s.confidence==0.0

def test_self_play_judgment_defaults():
    j=SelfPlayJudgment()
    assert j.overall==0.0
    assert j.corrected_answer is None

def test_self_play_no_keys():
    sp=SelfPlayDistill()
    rows=sp.run_sync("finance",n_problems=5)
    assert rows==[]

def test_self_play_init():
    sp=SelfPlayDistill(anthropic_key="fake",openai_key="fake",
                        claude_model="claude-opus-4-6",gpt_model="gpt-5.4")
    assert sp._ak=="fake"
    assert sp._ok=="fake"
    assert sp._thinking==True

def test_finance_bench_has_table_info():
    info=DOMAIN_BENCH_INFO["finance"]
    assert "table" in info["format"].lower() or "TABLE" in info["format"]
    assert "arithmetic" in info["bench"].lower() or "numerical" in info["bench"].lower()
