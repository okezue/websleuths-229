from __future__ import annotations
import os,copy
import torch
from transformers import (
    AutoModelForCausalLM,AutoTokenizer,
    TrainingArguments,Trainer,DataCollatorForLanguageModeling,
)
from peft import get_peft_model,LoraConfig,TaskType
from wm.types import TrainResult
from wm.cfg import TrainCfg
from wm.dream.core import Dreamer
from wm.dream.buffer import DreamBuffer
from wm.recipe.eatrd import EATRDRunner
from wm.recipe.dpmu import DPMURunner
from wm.recipe.eab_ssc import EABSSCRunner

class WMTrainer(Trainer):
    def __init__(self,dreamer:Dreamer|None=None,teacher=None,
                 dream_buf:DreamBuffer|None=None,total_steps:int=0,**kw):
        super().__init__(**kw)
        self._dreamer=dreamer
        self._teacher=teacher
        self._dbuf=dream_buf
        self._total=total_steps
        self._dream_loss_acc=0.0
        self._dream_steps=0
    def compute_loss(self,model,inputs,return_outputs=False,**kw):
        outputs=model(**inputs)
        loss=outputs.loss
        if self._dreamer and self._teacher is not None:
            dev=next(model.parameters()).device
            if next(self._teacher.parameters()).device!=dev:
                self._teacher=self._teacher.to(dev)
            d_inputs=self._dbuf.sample(dev) if self._dbuf else None
            if d_inputs is None:
                d_inputs=inputs
            with torch.no_grad():
                t_out=self._teacher(**d_inputs)
            s_out=model(**d_inputs) if d_inputs is not inputs else outputs
            dl=self._dreamer.step(
                self.state.global_step,self._total,
                s_out.logits,t_out.logits,
            )
            if dl is not None:
                loss=loss+dl
                self._dream_loss_acc+=dl.item()
                self._dream_steps+=1
        return (loss,outputs) if return_outputs else loss

class HFBack:
    def __init__(self,cfg:TrainCfg):
        self._cfg=cfg
        self._model=None
        self._tok=None
    def setup(self,model_name:str,lora_cfg:object,out_dir:str):
        self._tok=AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
        if self._tok.pad_token is None:
            self._tok.pad_token=self._tok.eos_token
        self._model=AutoModelForCausalLM.from_pretrained(model_name,trust_remote_code=True)
        lc=LoraConfig(
            r=self._cfg.lora.r,lora_alpha=self._cfg.lora.alpha,
            lora_dropout=self._cfg.lora.dropout,
            target_modules=self._cfg.lora.modules,
            task_type=TaskType.CAUSAL_LM,
        )
        self._model=get_peft_model(self._model,lc)
        self._out=out_dir
    def setup_from_model(self,model,tok,out_dir:str):
        self._model=model
        self._tok=tok
        if self._tok.pad_token is None:
            self._tok.pad_token=self._tok.eos_token
        self._out=out_dir
    def train(self,ds,dreamer:Dreamer|None=None,teacher=None,
              dream_prompts:list[str]|None=None,
              recipe:str|None=None,**kw)->TrainResult:
        if recipe:
            return self._train_recipe(ds,dream_prompts or [],recipe)
        return self._train_legacy(ds,dreamer,teacher,dream_prompts)
    def _train_recipe(self,ds,dream_prompts:list[str],recipe:str)->TrainResult:
        teacher=copy.deepcopy(self._model)
        teacher.eval()
        c=self._cfg
        if recipe=="eatrd":
            r=EATRDRunner(lr=c.lr,max_steps=c.max_steps,bs=c.bs,
                temp=c.dream.temp,max_len=512,
                dream_n=c.dream.n_dream,dream_len=c.dream.max_len)
            return r.run(self._model,teacher,ds,dream_prompts,self._tok)
        elif recipe=="dpmu":
            r=DPMURunner(lr=c.lr,max_steps=c.max_steps,bs=c.bs,
                temp=c.dream.temp,max_len=512,
                dream_n=c.dream.n_dream,dream_len=c.dream.max_len)
            return r.run(self._model,teacher,ds,dream_prompts,self._tok)
        elif recipe=="eab_ssc":
            r=EABSSCRunner(lr=c.lr,max_steps=c.max_steps,bs=c.bs,
                temp=c.dream.temp,dream_weight=c.dream.weight,max_len=512,
                dream_n=c.dream.n_dream,dream_len=c.dream.max_len)
            return r.day(self._model,teacher,ds,dream_prompts,self._tok)
        raise ValueError(f"unknown recipe: {recipe}")
    def _train_legacy(self,ds,dreamer,teacher,dream_prompts)->TrainResult:
        def tok_fn(ex):
            t=ex.get("text","")
            if not t:
                t=ex.get("prompt","")+"\n"+ex.get("completion","")
            return self._tok(t,truncation=True,max_length=512,padding=False)
        tok_ds=ds.map(tok_fn,remove_columns=[c for c in ds.column_names if c not in ("input_ids","attention_mask")])
        dc=DataCollatorForLanguageModeling(self._tok,mlm=False)
        ms=self._cfg.max_steps if self._cfg.max_steps>0 else int(len(tok_ds)/self._cfg.bs*self._cfg.epochs)
        args=TrainingArguments(
            output_dir=self._out,num_train_epochs=self._cfg.epochs,
            per_device_train_batch_size=self._cfg.bs,
            gradient_accumulation_steps=self._cfg.grad_acc,
            learning_rate=self._cfg.lr,
            warmup_steps=min(self._cfg.warmup,ms//2),
            fp16=self._cfg.fp16,bf16=self._cfg.bf16 and torch.cuda.is_available(),
            logging_steps=max(1,ms//10),save_strategy="no",
            max_steps=self._cfg.max_steps if self._cfg.max_steps>0 else -1,
            report_to="none",dataloader_pin_memory=False,
        )
        t_model=None
        if dreamer and teacher is None:
            t_model=copy.deepcopy(self._model);t_model.eval()
        elif teacher is not None:
            t_model=teacher
        dbuf=None
        if dreamer and dream_prompts:
            dcfg=self._cfg.dream
            dbuf=DreamBuffer(dream_prompts,self._tok,dcfg.max_len,dcfg.n_dream)
        trainer=WMTrainer(dreamer=dreamer,teacher=t_model,dream_buf=dbuf,
            total_steps=ms,model=self._model,args=args,
            train_dataset=tok_ds,data_collator=dc)
        res=trainer.train()
        dl=trainer._dream_loss_acc/max(1,trainer._dream_steps) if trainer._dream_steps>0 else None
        return TrainResult(loss=res.training_loss,steps=int(res.global_step),
            lr=self._cfg.lr,dream_loss=dl)
    def save(self,path:str):
        if self._model:
            self._model.save_pretrained(path)
            if self._tok:self._tok.save_pretrained(path)
    @property
    def model(self):return self._model
    @property
    def tokenizer(self):return self._tok
