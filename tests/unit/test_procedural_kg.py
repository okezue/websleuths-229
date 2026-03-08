import pytest
from wm.search.procedural_kg import (
    ProceduralKnowledge,procedural_to_train_rows,benchmark_q_to_train_rows)

def test_procedural_knowledge_defaults():
    pk=ProceduralKnowledge()
    assert pk.reasoning_chains==[]
    assert pk.worked_examples==[]
    assert pk.concepts==[]
    assert pk.claims==[]
    assert pk.qa_pairs==[]

def test_procedural_to_train_rows_reasoning():
    pk=ProceduralKnowledge(
        reasoning_chains=[{"title":"Calc P/E","steps":["Find price","Find EPS","Divide"]}])
    rows=procedural_to_train_rows(pk)
    assert len(rows)==1
    assert "Step 1" in rows[0]["text"]
    assert rows[0]["source"]=="procedural"
    assert rows[0]["authority"]==0.85

def test_procedural_to_train_rows_worked_examples():
    pk=ProceduralKnowledge(
        worked_examples=[{"question":"What is 2+2?","solution":"2+2=4"}])
    rows=procedural_to_train_rows(pk)
    assert len(rows)==1
    assert "Q:" in rows[0]["text"]
    assert "step by step" in rows[0]["text"]
    assert rows[0]["source"]=="procedural"

def test_procedural_to_train_rows_concepts():
    pk=ProceduralKnowledge(
        concepts=[{"term":"P/E Ratio","definition":"Price divided by earnings","example":"AAPL P/E=25"}])
    rows=procedural_to_train_rows(pk)
    assert len(rows)==1
    assert "P/E Ratio" in rows[0]["text"]
    assert "Example:" in rows[0]["text"]

def test_procedural_to_train_rows_claims():
    pk=ProceduralKnowledge(
        claims=[{"text":"SEC was founded in 1934","confidence":0.9}])
    rows=procedural_to_train_rows(pk)
    assert len(rows)==1
    assert rows[0]["source"]=="factual"
    assert rows[0]["authority"]==0.9

def test_procedural_to_train_rows_qa_pairs():
    pk=ProceduralKnowledge(
        qa_pairs=[{"question":"What is GDP?","answer":"Gross Domestic Product","explanation":"Total output"}])
    rows=procedural_to_train_rows(pk)
    assert len(rows)==1
    assert "Q:" in rows[0]["text"]

def test_procedural_to_train_rows_empty():
    pk=ProceduralKnowledge()
    rows=procedural_to_train_rows(pk)
    assert rows==[]

def test_procedural_to_train_rows_mixed():
    pk=ProceduralKnowledge(
        reasoning_chains=[{"title":"T","steps":["a","b"]}],
        worked_examples=[{"question":"Q","solution":"S"}],
        concepts=[{"term":"X","definition":"Y"}],
        claims=[{"text":"C","confidence":0.5}],
        qa_pairs=[{"question":"Q2","answer":"A2"}])
    rows=procedural_to_train_rows(pk)
    assert len(rows)==5
    sources=[r["source"] for r in rows]
    assert "procedural" in sources
    assert "factual" in sources

def test_benchmark_q_to_train_rows():
    qs=[{"question":"What is 1+1?","correct":"A","reasoning":"1+1=2","choices":["A) 2","B) 3"]}]
    rows=benchmark_q_to_train_rows(qs)
    assert len(rows)==1
    assert "step by step" in rows[0]["text"]
    assert rows[0]["source"]=="procedural"

def test_benchmark_q_to_train_rows_empty():
    rows=benchmark_q_to_train_rows([])
    assert rows==[]

def test_benchmark_q_no_reasoning():
    qs=[{"question":"What is 1+1?","correct":"A","choices":["A) 2","B) 3"]}]
    rows=benchmark_q_to_train_rows(qs)
    assert len(rows)==1
    assert "Answer: A" in rows[0]["text"]
