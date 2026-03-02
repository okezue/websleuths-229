#!/bin/bash
set -e
cd /home/ubuntu
source /opt/aws_neuronx_venv_pytorch_2_5_nxd_inference/bin/activate 2>/dev/null || source /opt/aws_neuron_venv_pytorch/bin/activate 2>/dev/null || true
pip install -q peft datasets tiktoken transformers accelerate optimum-neuron exa-py pydantic
git clone https://github.com/okezuebell/websleuths-229.git /home/ubuntu/wm || (cd /home/ubuntu/wm && git pull)
cd /home/ubuntu/wm
pip install -e ".[dev]" -q
export EXA_API_KEY="e337f35a-e56c-4ae7-8596-f44959053342"
python3 -c "
import torch
print('torch:',torch.__version__)
try:
    import torch_neuronx
    print('neuronx:',torch_neuronx.__version__)
except:
    print('neuronx: not found')
try:
    from optimum.neuron import NeuronModelForCausalLM
    print('optimum-neuron: OK')
except Exception as e:
    print('optimum-neuron:',e)
from wm.ingest import EpisodeIngestor
from wm.ingest.exa import ExaSrc
from wm.store import EpisodeStore
from wm.chunk import Chunker
from wm.dataset import DatasetBuilder
from wm.gate import EpisodeGate,exa_authority
from wm.recipe import EATRDRunner,DPMURunner
from wm.cfg import *
from wm.eval import Evaluator
from wm.registry import ModelRegistry
from wm.cert import Certifier
import os,copy
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import get_peft_model,LoraConfig,TaskType
print('=== INGEST ===')
src=ExaSrc()
eps=src.fetch('unsolved cold case evidence 2024',n=5)
eps=exa_authority(eps)
for e in eps:
    print(f'  [{e.eid[:8]}] auth={e.authority:.3f} {e.title[:50]}')
gate=EpisodeGate(GateCfg(min_sources=1,min_consistency=0.0,uncertainty_thresh=0.0))
gr=gate.check(eps)
print(f'gate: accept={gr.accept} sources={gr.n_sources} consistency={gr.consistency:.3f}')
store=EpisodeStore('/tmp/wm_ep.db')
store.put_many(eps)
print(f'store: {store.count()} episodes')
print('=== CHUNK + DATASET ===')
chunks=Chunker(ChunkCfg(max_tok=256,overlap=32)).chunk_many(store.all())
print(f'chunks: {len(chunks)}')
ds=DatasetBuilder(DatasetCfg(recipe=\"cpt\")).build(chunks)
print(f'dataset: {len(ds)} rows')
print('=== LOAD MODEL ===')
mn='meta-llama/Llama-3.2-1B'
tok=AutoTokenizer.from_pretrained(mn,trust_remote_code=True)
if tok.pad_token is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(mn,trust_remote_code=True,torch_dtype=torch.float32)
lc=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,target_modules=['q_proj','v_proj'],task_type=TaskType.CAUSAL_LM)
model=get_peft_model(model,lc)
model.print_trainable_parameters()
teacher=copy.deepcopy(model).eval()
P_DREAM=[
    'What is the capital of France?','Explain photosynthesis briefly.',
    'Who wrote Hamlet?','What causes earthquakes?',
    'How does the internet work?','What is DNA?',
    'Describe the water cycle.','What is machine learning?',
    'Who was Albert Einstein?','How do airplanes fly?',
]
print('=== RECIPE 1: EATRD ===')
r1=EATRDRunner(lr=2e-4,max_steps=20,bs=2,temp=2.0,eps_min=0.01,alpha=0.5,rho=0.01,lam_init=1.0,max_len=256,dream_n=4,dream_len=128)
res1=r1.run(model,teacher,ds,P_DREAM,tok)
print(f'  loss={res1.loss:.4f} dream={res1.dream_loss:.4f} steps={res1.steps} lambda={res1.extras[\"lambda\"]:.4f} eps_k={res1.extras[\"eps_k\"]:.4f}')
print('=== RECIPE 2: DPMU ===')
model2=copy.deepcopy(model)
teacher2=copy.deepcopy(teacher)
r2=DPMURunner(lr=2e-4,max_steps=20,bs=2,temp=2.0,n_dream_grads=1,max_len=256,dream_n=4,dream_len=128)
res2=r2.run(model2,teacher2,ds,P_DREAM,tok)
print(f'  loss={res2.loss:.4f} dream={res2.dream_loss:.4f} steps={res2.steps}')
print('=== EVAL ===')
ev=Evaluator(model,tok,max_len=128)
report=ev.evaluate(ds)
print(f'  ppl={report.ppl:.2f} acc={report.acc:.4f}')
print('=== REGISTRY + CERT ===')
os.makedirs('/tmp/wm_model',exist_ok=True)
model.save_pretrained('/tmp/wm_model')
tok.save_pretrained('/tmp/wm_model')
reg=ModelRegistry(RegCfg(local_dir='/tmp/wm_reg'))
v=reg.push('/tmp/wm_model',{'loss':res1.loss,'ppl':report.ppl,'acc':report.acc})
print(f'  pushed version {v.vid}')
cert=Certifier(CertCfg(min_acc=0.0,max_ppl_delta=1000.0,max_drift=1.0))
cr=cert.certify(report)
print(f'  certified: {cr.passed}')
store.close()
reg.close()
print('=== DONE ===')
"
