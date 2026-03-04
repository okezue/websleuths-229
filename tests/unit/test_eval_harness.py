import torch
from wm.bench.eval_harness import (
    BenchScore,_extract_letter,_extract_number,_num_close,
    _format_mcq,DomainEvalHarness,_MMLU_MAP,_DOMAIN_DS,
)

def test_benchscore_fields():
    bs=BenchScore(name="test",acc=0.75,n=100)
    assert bs.name=="test"
    assert bs.acc==0.75
    assert bs.n==100
    assert bs.extras=={}

def test_benchscore_extras():
    bs=BenchScore(name="x",acc=0.5,n=10,extras={"k":"v"})
    assert bs.extras["k"]=="v"

def test_extract_letter_direct():
    assert _extract_letter("A")=="A"
    assert _extract_letter("B) something")=="B"
    assert _extract_letter("C")=="C"
    assert _extract_letter("D is correct")=="D"

def test_extract_letter_embedded():
    assert _extract_letter("The answer is B")=="B"
    assert _extract_letter("I think C is right")=="C"

def test_extract_letter_empty():
    assert _extract_letter("")==""
    assert _extract_letter("no letter here")==""

def test_extract_number_simple():
    assert _extract_number("42")==42.0
    assert _extract_number("3.14")==3.14
    assert _extract_number("-5.0")==-5.0

def test_extract_number_with_prefix():
    assert _extract_number("$1,234.56")==1234.56
    assert _extract_number("about 99%")==99.0

def test_extract_number_none():
    assert _extract_number("no numbers")==None
    assert _extract_number("")==None

def test_num_close():
    assert _num_close(1.0,1.0)==True
    assert _num_close(1.005,1.0)==True
    assert _num_close(1.02,1.0)==False
    assert _num_close(0.0,0.0)==True
    assert _num_close(0.005,0.0)==True

def test_format_mcq():
    s=_format_mcq("What?",["opt1","opt2","opt3","opt4"])
    assert "Q: What?" in s
    assert "A) opt1" in s
    assert "B) opt2" in s
    assert "C) opt3" in s
    assert "D) opt4" in s
    assert s.endswith("Answer:")

def test_format_mcq_five():
    s=_format_mcq("Q",["a","b","c","d","e"])
    assert "E) e" in s

def test_domain_ds_keys():
    assert "finance" in _DOMAIN_DS
    assert "legal" in _DOMAIN_DS
    assert "chemistry" in _DOMAIN_DS
    assert "medicine" in _DOMAIN_DS

def test_mmlu_map_keys():
    assert _MMLU_MAP["finance"]=="professional_accounting"
    assert _MMLU_MAP["legal"]=="professional_law"
    assert _MMLU_MAP["chemistry"]=="college_chemistry"
    assert _MMLU_MAP["medicine"]=="professional_medicine"

def test_harness_init(tiny_model,tiny_tok):
    h=DomainEvalHarness(tiny_model,tiny_tok,n_samples=5)
    assert h._n==5
    assert h._ml==512

def test_harness_generate_examples(tiny_model,tiny_tok):
    h=DomainEvalHarness(tiny_model,tiny_tok,n_samples=5)
    exs=h.generate_examples("finance",n=2)
    assert len(exs)==2
    assert "prompt" in exs[0]
    assert "response" in exs[0]

def test_harness_eval_unknown_domain(tiny_model,tiny_tok):
    h=DomainEvalHarness(tiny_model,tiny_tok,n_samples=5)
    bs=h.eval_domain("nonexistent")
    assert bs.acc==0.0
    assert bs.n==0

def test_harness_eval_mmlu_unknown(tiny_model,tiny_tok):
    h=DomainEvalHarness(tiny_model,tiny_tok,n_samples=5)
    bs=h.eval_mmlu("nonexistent")
    assert bs.acc==0.0
    assert bs.n==0

def test_harness_eval_all_domains_empty(tiny_model,tiny_tok):
    h=DomainEvalHarness(tiny_model,tiny_tok,n_samples=2)
    r=h.eval_all_domains(["nonexistent"])
    assert "nonexistent" in r
    assert r["nonexistent"].n==0
