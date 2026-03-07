import copy,torch
from wm.dream.bank import DreamBank
from wm.dream.hdm import hard_dream_mine

def test_hdm_returns_topk(tiny_model,tiny_tok):
    b=DreamBank(tiny_tok,n=4,max_len=16)
    b.seed()
    teacher=copy.deepcopy(tiny_model).eval()
    results=hard_dream_mine(tiny_model,teacher,b,tiny_tok,
                            pool_n=20,topk=5,dev=torch.device("cpu"))
    assert len(results)==5
    for txt,bk,kl in results:
        assert isinstance(txt,str)
        assert isinstance(bk,str)
        assert isinstance(kl,float)

def test_hdm_sorted_desc(tiny_model,tiny_tok):
    b=DreamBank(tiny_tok,n=4,max_len=16)
    b.seed()
    teacher=copy.deepcopy(tiny_model).eval()
    results=hard_dream_mine(tiny_model,teacher,b,tiny_tok,
                            pool_n=15,topk=10,dev=torch.device("cpu"))
    kls=[r[2] for r in results]
    for i in range(len(kls)-1):
        assert kls[i]>=kls[i+1]

def test_hdm_empty_bank(tiny_model,tiny_tok):
    b=DreamBank(tiny_tok)
    teacher=copy.deepcopy(tiny_model).eval()
    results=hard_dream_mine(tiny_model,teacher,b,tiny_tok,
                            pool_n=10,topk=5)
    assert results==[]
