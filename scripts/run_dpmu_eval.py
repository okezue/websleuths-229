import os,copy,torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import get_peft_model,LoraConfig,TaskType
from wm.store import EpisodeStore
from wm.chunk import Chunker
from wm.dataset import DatasetBuilder
from wm.recipe import DPMURunner
from wm.eval import Evaluator
from wm.registry import ModelRegistry
from wm.cert import Certifier
from wm.cfg import *
print("=== LOAD ===")
mn="Qwen/Qwen2.5-1.5B"
tok=AutoTokenizer.from_pretrained(mn)
if tok.pad_token is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(mn,torch_dtype=torch.float32)
lc=LoraConfig(r=8,lora_alpha=16,lora_dropout=0.05,
    target_modules=["q_proj","v_proj"],task_type=TaskType.CAUSAL_LM)
model=get_peft_model(model,lc)
model.print_trainable_parameters()
teacher=copy.deepcopy(model).eval()
store=EpisodeStore("/tmp/wm_ep.db")
chunks=Chunker(ChunkCfg(max_tok=256,overlap=32)).chunk_many(store.all())
ds=DatasetBuilder(DatasetCfg(recipe="cpt")).build(chunks)
P_DREAM=[
    "What is the capital of France?","Explain photosynthesis briefly.",
    "Who wrote Hamlet?","What causes earthquakes?",
    "How does the internet work?","What is DNA?",
]
print("=== RECIPE 2: DPMU (r=8, bs=1, 10 steps) ===")
r2=DPMURunner(lr=2e-4,max_steps=10,bs=1,temp=2.0,
    n_dream_grads=1,max_len=128,dream_n=2,dream_len=64)
res2=r2.run(model,teacher,ds,P_DREAM,tok)
print(f"  loss={res2.loss:.4f} dream={res2.dream_loss:.4f} steps={res2.steps}")
del teacher
print("=== EVAL ===")
ev=Evaluator(model,tok,max_len=128)
report=ev.evaluate(ds)
print(f"  ppl={report.ppl:.2f} acc={report.acc:.4f}")
print("=== REGISTRY + CERT ===")
os.makedirs("/tmp/wm_model2",exist_ok=True)
model.save_pretrained("/tmp/wm_model2")
tok.save_pretrained("/tmp/wm_model2")
reg=ModelRegistry(RegCfg(local_dir="/tmp/wm_reg"))
v=reg.push("/tmp/wm_model2",{"loss":res2.loss,"ppl":report.ppl,"acc":report.acc})
print(f"  pushed version {v.vid}")
cert=Certifier(CertCfg(min_acc=0.0,max_ppl_delta=1000.0,max_drift=1.0))
cr=cert.certify(report)
print(f"  certified: {cr.passed}")
for c in cr.checks:
    print(f"    {c.name}: passed={c.passed} val={c.val:.4f} thresh={c.thresh:.4f}")
store.close();reg.close()
print("=== DONE ===")
