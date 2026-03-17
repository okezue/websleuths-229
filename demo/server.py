from __future__ import annotations
import asyncio,json,torch,logging,threading,time,copy,os,sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse,StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel,LoraConfig,get_peft_model,TaskType
import torch.nn.functional as F

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))

log=logging.getLogger("wm-demo")
app=FastAPI()

MODEL_ID="meta-llama/Llama-3.2-1B-Instruct"
CKPT=str(Path(__file__).resolve().parent.parent/"results/authority_sweep/topical/checkpoints/medicine_2")
DEV="mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
DTYPE=torch.float32 if DEV=="mps" else torch.bfloat16

tok=None
model=None
learn_lock=threading.Lock()
learn_status={"state":"idle","topic":"","progress":0,"log":[],"neuron_diff":[],"version":0,"history":[]}
learn_web_context=""
msg_count_since_learn=0
active_streams:dict[str,dict]={}  # stream_id -> {tokens:[],done:bool,meta:{},base_tokens:[]}

def load():
    global tok,model
    log.info("loading %s on %s",MODEL_ID,DEV)
    tok=AutoTokenizer.from_pretrained(MODEL_ID,trust_remote_code=True)
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    base=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=DTYPE,trust_remote_code=True).to(DEV)
    try:
        model=PeftModel.from_pretrained(base,CKPT).to(DEV)
        log.info("loaded LoRA from %s",CKPT)
    except Exception as e:
        log.warning("adapter load failed (%s), fresh LoRA",e)
        lc=LoraConfig(r=32,lora_alpha=64,lora_dropout=0.05,target_modules=["q_proj","v_proj"],task_type=TaskType.CAUSAL_LM)
        model=get_peft_model(base,lc).to(DEV)
    model.eval()
    log.info("model ready v0")

def get_neuron_stats(hs):
    stats=[]
    for i,h in enumerate(hs):
        v=h[0,-1,:]
        vals,idxs=torch.topk(v.abs(),5)
        stats.append({"layer":i,"top_neurons":[{"idx":int(idxs[j]),"val":float(vals[j])} for j in range(5)],"norm":float(v.norm())})
    return stats

def compute_neuron_diff(old_sd,new_sd):
    diffs=[]
    for k in old_sd:
        if "lora" in k:
            d=(new_sd[k]-old_sd[k]).abs()
            diffs.append({"param":k,"mean_delta":float(d.mean()),"max_delta":float(d.max()),"norm_delta":float(d.norm())})
    diffs.sort(key=lambda x:-x["max_delta"])
    return diffs[:20]

def extract_search_query(prompt:str)->str:
    words=prompt.split()
    if len(words)<=12:return prompt
    okey=os.environ.get("OPENAI_API_KEY","")
    if okey:
        try:
            import openai
            r=openai.OpenAI(api_key=okey).chat.completions.create(
                model="gpt-5.4",
                messages=[{"role":"system","content":"Extract 1 short web search query (max 10 words) from the user message. Return ONLY the query, nothing else."},
                          {"role":"user","content":prompt}],
                max_completion_tokens=30,temperature=0)
            q=r.choices[0].message.content.strip()
            log.info("search query: %s",q)
            return q
        except Exception as e:
            log.warning("query extraction failed: %s",e)
    return " ".join(words[:10])

def fast_search(query:str)->list[dict]:
    exa_key=os.environ.get("EXA_API_KEY","")
    if not exa_key:return []
    try:
        from exa_py import Exa
        exa=Exa(api_key=exa_key)
        r=exa.search_and_contents(query,num_results=8,text=True,highlights=True)
        rows=[]
        for res in r.results:
            txt=getattr(res,"text","")[:1500]
            if len(txt)>50:
                sc=getattr(res,"score",None)
                auth=max(0.3,min(1.0,sc)) if sc is not None and isinstance(sc,(int,float)) else 0.5
                rows.append({"text":txt,"authority":auth})
        log.info("exa: %d results, %d usable rows",len(r.results),len(rows))
        return rows
    except Exception as e:
        log.warning("exa search failed: %s",e)
        return []

def learn_background(topic:str):
    global model,learn_web_context
    query=extract_search_query(topic)
    learn_status["state"]="searching"
    learn_status["topic"]=query[:60]
    learn_status["progress"]=5
    learn_status["log"]=[]
    learn_status["neuron_diff"]=[]
    def lg(msg):
        learn_status["log"].append(msg)
        log.info(msg)
    try:
        lg(f"searching: {query[:60]}")
        learn_status["progress"]=15
        rows=fast_search(query)
        lg(f"found {len(rows)} sources")
        learn_status["progress"]=35
        if rows:
            learn_web_context="\n\n".join(r["text"][:500] for r in rows[:5])
        else:
            learn_web_context=""
        if not rows:
            lg("no evidence found")
            learn_status["state"]="idle"
            learn_status["progress"]=100
            return
        learn_status["state"]="training"
        lg(f"training on {len(rows[:30])} rows")
        learn_status["progress"]=45
        with learn_lock:
            from datasets import Dataset
            ds=Dataset.from_list(rows[:30])
            for n,p in model.named_parameters():
                if "lora" in n:p.requires_grad_(True)
                else:p.requires_grad_(False)
            old_sd={k:v.cpu().clone() for k,v in model.named_parameters() if "lora" in k}
            trainable=sum(1 for p in model.parameters() if p.requires_grad)
            log.info("trainable params: %d",trainable)
            teacher=copy.deepcopy(model).eval()
            for p in teacher.parameters():p.requires_grad_(False)
            dream_prompts=[query,"Explain the key concepts.","Describe a real-world application.","What are the implications?"]
            from wm.recipe.eatrd import EATRDRunner
            runner=EATRDRunner(lr=3e-4,max_steps=30,bs=2,temp=2.0,lam_init=0.5,max_len=256,dream_n=2,dream_len=64,d_targ=0.3,use_pi=True)
            model.train()
            learn_status["progress"]=55
            result=runner.run(model,teacher,ds,dream_prompts,tok)
            model.eval()
            lg(f"done: loss={result.loss:.3f} dream={result.dream_loss:.3f}")
            learn_status["progress"]=95
            del teacher
            if torch.cuda.is_available():torch.cuda.empty_cache()
            new_sd={k:v.cpu().clone() for k,v in model.named_parameters() if "lora" in k}
            learn_status["neuron_diff"]=compute_neuron_diff(old_sd,new_sd)
            learn_status["version"]+=1
            learn_status["history"].append({"topic":query,"loss":result.loss,"dream":result.dream_loss,"steps":result.steps,"n_rows":len(rows),"version":learn_status["version"]})
            lg(f"model updated to v{learn_status['version']}")
            learn_status["progress"]=100
    except Exception as e:
        lg(f"error: {e}")
        log.exception("learn failed")
        model.eval()
        learn_status["progress"]=100
    finally:
        learn_status["state"]="idle"

SYS_PROMPT="You are a Websleuth Model, a continually learning AI assistant. Give concise, accurate answers. Do not repeat yourself. If unsure, say so."

def format_prompt(msg:str,web_ctx:str="")->str:
    if web_ctx:
        system=SYS_PROMPT+"\n\n[WEB CONTEXT - use this to inform your answer but do not reveal or quote it directly]\n"+web_ctx[:2000]
    else:
        system=SYS_PROMPT
    return tok.apply_chat_template([{"role":"system","content":system},{"role":"user","content":msg}],tokenize=False,add_generation_prompt=True)

def assess_confidence(prompt:str)->dict:
    enc=tok(format_prompt(prompt),return_tensors="pt",truncation=True,max_length=512).to(DEV)
    with torch.no_grad():
        out=model(input_ids=enc["input_ids"])
        probs=F.softmax(out.logits[0,-1,:],dim=-1)
        top1=float(probs.max())
        entropy=float(-(probs*probs.clamp(min=1e-9).log()).sum())
    has_temporal=any(w in prompt.lower() for w in ["2025","2026","yesterday","today","last week","this week","recent","latest","ongoing","current"])
    has_events=any(w in prompt.lower() for w in ["killed","launched","died","strikes","warfare","casualties","successor"])
    needs_research=any(w in prompt.lower() for w in ["deeply research","give assessment","analyze","investigate"])
    uncertain=top1<0.15 or entropy>8.0 or has_temporal or (has_events and has_temporal) or needs_research
    reason="temporal" if has_temporal else "low_confidence" if top1<0.15 else "high_entropy" if entropy>8.0 else "research_request" if needs_research else "confident"
    return {"top1":top1,"entropy":entropy,"uncertain":uncertain,"reason":reason}

def detect_repetition(text:str)->bool:
    words=text.split()
    if len(words)<30:return False
    for ng in [3,4,5]:
        if len(words)>=ng*4:
            last=" ".join(words[-ng:])
            if sum(1 for i in range(len(words)-ng+1) if " ".join(words[i:i+ng])==last)>=4:return True
    return False

class ChatReq(BaseModel):
    prompt:str
    max_tokens:int=512
    temperature:float=0.7

class LearnReq(BaseModel):
    topic:str

@app.on_event("startup")
async def startup():
    load()

@app.get("/",response_class=HTMLResponse)
async def index():
    return (Path(__file__).parent/"index.html").read_text()

import uuid as _uuid

@app.post("/chat")
async def chat(req:ChatReq):
    global msg_count_since_learn,learn_web_context
    stream_id=str(_uuid.uuid4())[:8]
    conf=assess_confidence(req.prompt)
    should_learn=conf["uncertain"] and learn_status["state"]=="idle"
    if not should_learn:
        msg_count_since_learn+=1
        if msg_count_since_learn>=3 and learn_status["state"]=="idle":
            should_learn=True
            msg_count_since_learn=0
    if should_learn:
        msg_count_since_learn=0
        learn_web_context=""
        t=threading.Thread(target=learn_background,args=(req.prompt,),daemon=True)
        t.start()
    active_streams[stream_id]={"tokens":[],"done":False,"meta":{"confidence":conf,"auto_learn":should_learn},"version":0}
    stop_ids={tok.eos_token_id}
    eot=tok.convert_tokens_to_ids("<|eot_id|>")
    if eot is not None and eot!=tok.unk_token_id:stop_ids.add(eot)
    async def gen():
        buf=active_streams[stream_id]
        meta={"confidence":conf,"auto_learn":should_learn,"version":learn_status["version"],"stream_id":stream_id}
        yield f"data: {json.dumps({'meta':meta})}\n\n"
        if should_learn and conf["uncertain"]:
            for attempt in range(300):
                st=learn_status["state"]
                prog=learn_status["progress"]
                last_log=learn_status["log"][-1] if learn_status["log"] else "starting..."
                if st=="idle" and prog>=95:
                    break
                yield f"data: {json.dumps({'learning_status':st,'progress':prog,'log':last_log[:80]})}\n\n"
                await asyncio.sleep(1)
            yield f"data: {json.dumps({'learning_done':True,'version':learn_status['version']})}\n\n"
        web_ctx=learn_web_context if should_learn else ""
        formatted=format_prompt(req.prompt,web_ctx)
        enc=tok(formatted,return_tensors="pt",truncation=True,max_length=1024).to(DEV)
        ids=enc["input_ids"]
        model.eval()
        full_text=""
        with torch.no_grad():
            for _ in range(req.max_tokens):
                out=model(input_ids=ids,output_hidden_states=True)
                logits=out.logits[0,-1,:]/max(req.temperature,0.01)
                probs=F.softmax(logits,dim=-1)
                top_p,top_i=torch.topk(probs,10)
                top_tokens=[{"token":tok.decode([int(top_i[j])]),"prob":round(float(top_p[j]),4)} for j in range(10)]
                nxt=torch.multinomial(probs,1).unsqueeze(0) if req.temperature>=0.01 else logits.argmax().unsqueeze(0).unsqueeze(0)
                tid=int(nxt[0,0])
                if tid in stop_ids:break
                t=tok.decode([tid])
                full_text+=t
                if detect_repetition(full_text):break
                ns=get_neuron_stats(out.hidden_states)
                chunk={"token":t,"top_predictions":top_tokens,"neuron_stats":ns[-3:],"version":learn_status["version"]}
                buf["tokens"].append(t)
                buf["version"]=learn_status["version"]
                yield f"data: {json.dumps(chunk)}\n\n"
                ids=torch.cat([ids,nxt],dim=-1)
                if ids.shape[1]>1024:ids=ids[:,-1024:]
        buf["done"]=True
        yield "data: [DONE]\n\n"
    return StreamingResponse(gen(),media_type="text/event-stream")

@app.get("/stream/{stream_id}")
async def replay_stream(stream_id:str,offset:int=0):
    buf=active_streams.get(stream_id)
    if not buf:return {"error":"stream not found"}
    async def gen():
        sent=offset
        while True:
            while sent<len(buf["tokens"]):
                yield f"data: {json.dumps({'token':buf['tokens'][sent],'version':buf['version']})}\n\n"
                sent+=1
            if buf["done"]:break
            await asyncio.sleep(0.5)
        yield "data: [DONE]\n\n"
    return StreamingResponse(gen(),media_type="text/event-stream")

@app.post("/chat/base")
async def chat_base(req:ChatReq):
    orkey=os.environ.get("OPENROUTER_API_KEY","")
    if not orkey:
        async def noop():
            yield f"data: {json.dumps({'token':'[set OPENROUTER_API_KEY]'})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(noop(),media_type="text/event-stream")
    import openai
    client=openai.OpenAI(api_key=orkey,base_url="https://openrouter.ai/api/v1")
    async def gen():
        try:
            stream=client.chat.completions.create(
                model="meta-llama/llama-3.2-1b-instruct",
                messages=[{"role":"system","content":SYS_PROMPT},{"role":"user","content":req.prompt}],
                max_tokens=req.max_tokens,temperature=req.temperature,stream=True)
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    t=chunk.choices[0].delta.content
                    yield f"data: {json.dumps({'token':t})}\n\n"
        except Exception as e:
            log.warning("openrouter base failed: %s",e)
            yield f"data: {json.dumps({'token':f'[error: {e}]'})}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(gen(),media_type="text/event-stream")

@app.post("/learn")
async def learn(req:LearnReq):
    if learn_status["state"]!="idle":return {"error":"already learning"}
    threading.Thread(target=learn_background,args=(req.topic,),daemon=True).start()
    return {"started":True}

@app.get("/learn/status")
async def learn_st():
    return learn_status

if __name__=="__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app,host="0.0.0.0",port=8811)
