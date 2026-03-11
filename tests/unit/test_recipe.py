import torch,copy
from datasets import Dataset
from wm.recipe.base import weighted_ce,dream_kl,multi_temp_dream_kl,evidence_eps,grad_vec,set_grad,WeightedLMCollator
from wm.recipe.eatrd import EATRDRunner
from wm.recipe.dpmu import DPMURunner,_project,_project_multi
from wm.recipe.eab_ssc import EABSSCRunner,AdapterBank,SleepConsolidator,_find_lora_pairs,_svd_trunc,_spectral_reg
from wm.dream.buffer import DreamBuffer
from wm.dream.bank import DreamBank
from wm.cfg import TrainCfg

_DP=["What is gravity?","How does rain form?","Who built the pyramids?",
     "What is photosynthesis?","Explain quantum mechanics."]*2

def _ds():
    return Dataset.from_dict({
        "text":["hello world this is test data for training "*5]*20,
        "authority":[0.5+0.02*i for i in range(20)],
    })

def test_weighted_ce():
    B,S,V=2,8,32
    logits=torch.randn(B,S,V)
    ids=torch.randint(0,V,(B,S))
    mask=torch.ones(B,S)
    w=torch.tensor([1.0,0.5])
    l=weighted_ce(logits,ids,mask,w)
    assert l.shape==()
    assert l.item()>0

def test_dream_kl():
    s=torch.randn(2,4,32)
    t=torch.randn(2,4,32)
    l=dream_kl(s,t,temp=2.0)
    assert l.item()>=0

def test_dream_kl_identical():
    x=torch.randn(2,4,32)
    l=dream_kl(x,x,temp=2.0)
    assert l.item()>=0
    assert l.item()<0.01

def test_dream_kl_matches_multi_temp_uniform():
    s=torch.randn(2,4,32)
    t=torch.randn(2,4,32)
    l1=dream_kl(s,t,temp=2.0)
    l2=multi_temp_dream_kl(s,t,[2.0,2.0])
    assert torch.allclose(l1,l2,atol=1e-6,rtol=1e-5)

def test_evidence_eps():
    w=torch.tensor([0.8,0.9,0.7])
    e=evidence_eps(w,3,eps_min=0.01,alpha=0.5)
    assert e>0.01
    e2=evidence_eps(torch.tensor([0.1]),1,eps_min=0.01,alpha=0.5)
    assert e>e2

def test_grad_vec(tiny_model):
    x=torch.randint(0,256,(1,4))
    out=tiny_model(x,labels=x)
    out.loss.backward()
    g=grad_vec(tiny_model)
    assert g.numel()>0
    tiny_model.zero_grad()

def test_project():
    g_ep=torch.tensor([1.0,-1.0,0.0])
    g_dr=torch.tensor([0.0,1.0,0.0])
    g=_project(g_ep,g_dr)
    assert (g_dr*g).sum()>=0

def test_project_no_conflict():
    g_ep=torch.tensor([1.0,1.0])
    g_dr=torch.tensor([1.0,0.0])
    g=_project(g_ep,g_dr)
    assert torch.allclose(g,g_ep)

def test_project_multi():
    g_ep=torch.tensor([1.0,-1.0,-1.0])
    G=[torch.tensor([0.0,1.0,0.0]),torch.tensor([0.0,0.0,1.0])]
    g=_project_multi(g_ep,G)
    for gd in G:
        assert (gd*g).sum()>=-0.1

def test_eatrd_runner(tiny_model,tiny_tok,tmp_dir):
    ds=_ds()
    teacher=copy.deepcopy(tiny_model).eval()
    r=EATRDRunner(lr=1e-3,max_steps=4,bs=2,temp=2.0,
        max_len=32,dream_n=2,dream_len=16)
    res=r.run(tiny_model,teacher,ds,_DP,tiny_tok)
    assert res.steps==4
    assert res.loss>0
    assert res.dream_loss is not None
    assert "lambda" in res.extras
    assert "eps_k" in res.extras

def test_dpmu_runner(tiny_model,tiny_tok,tmp_dir):
    ds=_ds()
    teacher=copy.deepcopy(tiny_model).eval()
    r=DPMURunner(lr=1e-3,max_steps=4,bs=2,temp=2.0,
        max_len=32,dream_n=2,dream_len=16)
    res=r.run(tiny_model,teacher,ds,_DP,tiny_tok)
    assert res.steps==4
    assert res.loss>0

def test_dpmu_multi_constraint(tiny_model,tiny_tok):
    ds=_ds()
    teacher=copy.deepcopy(tiny_model).eval()
    r=DPMURunner(lr=1e-3,max_steps=2,bs=2,n_dream_grads=3,
        max_len=32,dream_n=2,dream_len=16)
    res=r.run(tiny_model,teacher,ds,_DP,tiny_tok)
    assert res.steps==2

def test_eab_ssc_day(tiny_model,tiny_tok):
    ds=_ds()
    teacher=copy.deepcopy(tiny_model).eval()
    r=EABSSCRunner(lr=1e-3,max_steps=4,bs=2,temp=2.0,
        max_len=32,dream_n=2,dream_len=16)
    res=r.day(tiny_model,teacher,ds,_DP,tiny_tok)
    assert res.steps==4
    assert res.dream_loss is not None

def test_adapter_bank(tiny_model):
    bank=AdapterBank()
    bank.add(tiny_model,weight=0.8)
    bank.add(tiny_model,weight=1.0)
    assert bank.size==2
    all_a=bank.get_all()
    assert len(all_a)==2
    assert all_a[0][0]==0.8

def test_svd_trunc():
    dw=torch.randn(16,32)
    A,B=_svd_trunc(dw,r=4)
    assert A.shape==(4,32)
    assert B.shape==(16,4)
    recon=B@A
    err=(recon-dw).norm()/dw.norm()
    assert err<1.0

def test_hfback_recipe(tiny_model,tiny_tok,tmp_dir):
    from wm.train.hf import HFBack
    cfg=TrainCfg(backend="hf",epochs=1,bs=2,max_steps=4,lr=1e-3,
        warmup=1,bf16=False,fp16=False,grad_acc=1)
    b=HFBack(cfg)
    b.setup_from_model(tiny_model,tiny_tok,tmp_dir)
    res=b.train(_ds(),recipe="eatrd",dream_prompts=_DP)
    assert res.steps==4
    assert res.dream_loss is not None

def test_hfback_dpmu(tiny_model,tiny_tok,tmp_dir):
    from wm.train.hf import HFBack
    cfg=TrainCfg(backend="hf",epochs=1,bs=2,max_steps=4,lr=1e-3,
        warmup=1,bf16=False,fp16=False,grad_acc=1)
    b=HFBack(cfg)
    b.setup_from_model(tiny_model,tiny_tok,tmp_dir)
    res=b.train(_ds(),recipe="dpmu",dream_prompts=_DP)
    assert res.steps==4

def test_multi_temp_dream_kl_shape():
    s=torch.randn(3,4,32)
    t=torch.randn(3,4,32)
    l=multi_temp_dream_kl(s,t,[1.0,2.0,1.5])
    assert l.shape==()
    assert l.item()>=0

def test_multi_temp_uniform_consistent():
    s=torch.randn(2,4,32)
    t=torch.randn(2,4,32)
    l1=multi_temp_dream_kl(s,t,[2.0,2.0])
    l2=multi_temp_dream_kl(s,t,[2.0,2.0])
    assert abs(l1.item()-l2.item())<1e-5
    l_lo=multi_temp_dream_kl(s,t,[1.0,1.0])
    assert l_lo.item()>=0

def test_eatrd_pi_lambda_bounded(tiny_model,tiny_tok):
    ds=_ds()
    teacher=copy.deepcopy(tiny_model).eval()
    r=EATRDRunner(lr=1e-3,max_steps=10,bs=2,temp=2.0,
        max_len=32,dream_n=2,dream_len=16,
        d_targ=0.5,lam_floor=0.01,lam_ceil=10.0,use_pi=True)
    res=r.run(tiny_model,teacher,ds,_DP,tiny_tok)
    lam=res.extras["lambda"]
    assert 0.01<=lam<=10.0
    assert res.extras["use_pi"]==True
    for h in res.history:
        assert 0.01<=h["lambda"]<=10.0

def test_eatrd_backward_compat(tiny_model,tiny_tok):
    ds=_ds()
    teacher=copy.deepcopy(tiny_model).eval()
    r=EATRDRunner(lr=1e-3,max_steps=4,bs=2,temp=2.0,
        max_len=32,dream_n=2,dream_len=16,use_pi=False)
    res=r.run(tiny_model,teacher,ds,_DP,tiny_tok)
    assert res.steps==4
    assert "lambda" in res.extras

def test_eatrd_with_dbank(tiny_model,tiny_tok):
    ds=_ds()
    teacher=copy.deepcopy(tiny_model).eval()
    b=DreamBank(tiny_tok,n=2,max_len=16)
    b.seed()
    r=EATRDRunner(lr=1e-3,max_steps=4,bs=2,temp=2.0,
        max_len=32,dream_n=2,dream_len=16)
    res=r.run(tiny_model,teacher,ds,_DP,tiny_tok,dbank=b)
    assert res.steps==4
    assert res.dream_loss is not None

def test_dpmu_lite_cached(tiny_model,tiny_tok):
    ds=_ds()
    teacher=copy.deepcopy(tiny_model).eval()
    r=DPMURunner(lr=1e-3,max_steps=8,bs=2,temp=2.0,
        max_len=32,dream_n=2,dream_len=16,
        grad_refresh_k=3,grad_ema_decay=0.9)
    res=r.run(tiny_model,teacher,ds,_DP,tiny_tok)
    assert res.steps==8
    cached_count=sum(1 for h in res.history if h.get("cached",False))
    assert cached_count>0

def test_dpmu_with_dbank(tiny_model,tiny_tok):
    ds=_ds()
    teacher=copy.deepcopy(tiny_model).eval()
    b=DreamBank(tiny_tok,n=2,max_len=16)
    b.seed()
    r=DPMURunner(lr=1e-3,max_steps=4,bs=2,temp=2.0,
        max_len=32,dream_n=2,dream_len=16)
    res=r.run(tiny_model,teacher,ds,_DP,tiny_tok,dbank=b)
    assert res.steps==4

def test_eab_ssc_with_dbank(tiny_model,tiny_tok):
    ds=_ds()
    teacher=copy.deepcopy(tiny_model).eval()
    b=DreamBank(tiny_tok,n=2,max_len=16)
    b.seed()
    r=EABSSCRunner(lr=1e-3,max_steps=4,bs=2,temp=2.0,
        max_len=32,dream_n=2,dream_len=16)
    res=r.day(tiny_model,teacher,ds,_DP,tiny_tok,dbank=b)
    assert res.steps==4
    assert res.dream_loss is not None
