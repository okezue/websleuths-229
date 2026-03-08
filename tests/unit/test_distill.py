import pytest
from wm.distill.claude_distill import (
    DistillQuestion,DistillAnswer,build_distill_rows,BENCH_DOMAINS)
from wm.distill.claude_judge import JudgeResult,build_judge_rows

def test_distill_question_defaults():
    dq=DistillQuestion()
    assert dq.question==""
    assert dq.choices==[]
    assert dq.correct==""

def test_distill_answer_defaults():
    da=DistillAnswer()
    assert da.reasoning==""
    assert da.answer==""
    assert da.confidence==0.0

def test_bench_domains():
    assert "finance" in BENCH_DOMAINS
    assert "legal" in BENCH_DOMAINS
    assert "chemistry" in BENCH_DOMAINS
    assert "medicine" in BENCH_DOMAINS
    for d in BENCH_DOMAINS.values():
        assert "dataset" in d
        assert "style" in d
        assert "examples" in d

def test_build_distill_rows_with_reasoning():
    qs=[DistillQuestion(question="What is P/E?",choices=["A) 10","B) 20"],correct="A")]
    ans=[DistillAnswer(question="What is P/E?",choices=["A) 10","B) 20"],
                        reasoning="P/E = price/earnings = 50/5 = 10",
                        answer="A",confidence=0.9)]
    rows=build_distill_rows(qs,ans)
    assert len(rows)==2
    assert rows[0]["source"]=="distill"
    assert rows[0]["authority"]==0.95
    assert "step by step" in rows[0]["text"]
    assert rows[1]["authority"]==0.9

def test_build_distill_rows_no_reasoning():
    qs=[DistillQuestion(question="Q",choices=["A","B"],correct="A")]
    ans=[DistillAnswer(question="Q",choices=["A","B"],answer="A")]
    rows=build_distill_rows(qs,ans)
    assert len(rows)==1
    assert "Answer: A" in rows[0]["text"]

def test_build_distill_rows_empty():
    rows=build_distill_rows([],[])
    assert rows==[]

def test_judge_result_defaults():
    jr=JudgeResult()
    assert jr.correctness==0.0
    assert jr.reasoning_quality==0.0
    assert jr.overall==0.0

def test_build_judge_rows_high_score():
    jrs=[JudgeResult(question="Q",model_response="good answer",gold="gold",
                      overall=0.8)]
    rows=build_judge_rows(jrs)
    assert len(rows)==1
    assert rows[0]["authority"]==0.8
    assert rows[0]["source"]=="judge"
    assert "good answer" in rows[0]["text"]

def test_build_judge_rows_low_score_with_claude():
    jrs=[JudgeResult(question="Q",model_response="bad",gold="gold",overall=0.2)]
    claude_ans=["Claude's answer"]
    rows=build_judge_rows(jrs,claude_ans)
    assert len(rows)==1
    assert "Claude's answer" in rows[0]["text"]
    assert rows[0]["authority"]==0.9

def test_build_judge_rows_low_score_no_claude():
    jrs=[JudgeResult(question="Q",model_response="bad",gold="gold",overall=0.2)]
    rows=build_judge_rows(jrs)
    assert len(rows)==1
    assert "gold" in rows[0]["text"]
    assert rows[0]["authority"]==0.8

def test_build_judge_rows_empty():
    rows=build_judge_rows([])
    assert rows==[]

def test_build_judge_rows_mixed():
    jrs=[
        JudgeResult(question="Q1",model_response="good",gold="g1",overall=0.9),
        JudgeResult(question="Q2",model_response="bad",gold="g2",overall=0.3),
    ]
    claude_ans=["ca1","ca2"]
    rows=build_judge_rows(jrs,claude_ans)
    assert len(rows)==2
    assert rows[0]["authority"]==0.9
    assert "good" in rows[0]["text"]
    assert "ca2" in rows[1]["text"]
