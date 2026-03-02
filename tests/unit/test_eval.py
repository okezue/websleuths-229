from datasets import Dataset
from wm.eval import Evaluator

def _ds():
    return Dataset.from_dict({"text":["hello world test data "*10]*5})

def test_ppl(tiny_model,tiny_tok):
    ev=Evaluator(tiny_model,tiny_tok,max_len=64)
    r=ev.evaluate(_ds())
    assert r.ppl>0
    assert r.acc>=0

def test_acc_range(tiny_model,tiny_tok):
    ev=Evaluator(tiny_model,tiny_tok,max_len=64)
    r=ev.evaluate(_ds())
    assert 0<=r.acc<=1

def test_empty_ds(tiny_model,tiny_tok):
    ev=Evaluator(tiny_model,tiny_tok)
    ds=Dataset.from_dict({"text":[]})
    r=ev.evaluate(ds)
    assert r.ppl>0

def test_qa_f1(tiny_model,tiny_tok):
    ds=Dataset.from_dict({
        "prompt":["What is this?"]*3,
        "completion":["hello"]*3,
        "text":["hello world test"]*3,
    })
    ev=Evaluator(tiny_model,tiny_tok,max_len=32)
    r=ev.evaluate(ds,do_qa=True)
    assert 0<=r.qa_f1<=1
    assert r.ppl>0
