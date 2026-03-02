from __future__ import annotations
from wm.types import TrainResult
from wm.cfg import TrainCfg

class NeuronBack:
    def __init__(self,cfg:TrainCfg|None=None):
        self._cfg=cfg or TrainCfg()
        self._available=False
        self._model=None
        self._tok=None
        try:
            from optimum.neuron import NeuronModelForCausalLM
            self._available=True
        except ImportError:
            pass
    @property
    def available(self)->bool:
        return self._available
    def setup(self,model_name:str,lora_cfg:object,out_dir:str):
        if not self._available:
            raise RuntimeError("optimum-neuron not installed")
        from optimum.neuron import NeuronModelForCausalLM
        from transformers import AutoTokenizer
        from peft import LoraConfig,TaskType,get_peft_model
        self._tok=AutoTokenizer.from_pretrained(model_name)
        if self._tok.pad_token is None:
            self._tok.pad_token=self._tok.eos_token
        self._model=NeuronModelForCausalLM.from_pretrained(
            model_name,export=True,
            batch_size=self._cfg.bs,
            sequence_length=512,
            auto_cast_type="bf16" if self._cfg.bf16 else "fp32",
        )
        c=self._cfg
        lc=LoraConfig(r=c.lora.r,lora_alpha=c.lora.alpha,
            lora_dropout=c.lora.dropout,
            target_modules=c.lora.modules,
            task_type=TaskType.CAUSAL_LM)
        self._model=get_peft_model(self._model,lc)
        self._out=out_dir
    def train(self,ds,**kw)->TrainResult:
        if not self._available:
            raise RuntimeError("optimum-neuron not installed")
        from optimum.neuron import NeuronSFTTrainer
        from transformers import TrainingArguments
        c=self._cfg
        ms=c.max_steps if c.max_steps>0 else -1
        args=TrainingArguments(
            output_dir=self._out,num_train_epochs=c.epochs,
            per_device_train_batch_size=c.bs,
            gradient_accumulation_steps=c.grad_acc,
            learning_rate=c.lr,bf16=c.bf16,
            save_strategy="no",max_steps=ms,report_to="none",
        )
        trainer=NeuronSFTTrainer(
            model=self._model,args=args,
            train_dataset=ds,tokenizer=self._tok)
        res=trainer.train()
        return TrainResult(loss=res.training_loss,
            steps=int(res.global_step),lr=c.lr)
    def save(self,path:str):
        if not self._available:
            raise RuntimeError("optimum-neuron not installed")
        if self._model:
            self._model.save_pretrained(path)
    @property
    def model(self):return self._model
    @property
    def tokenizer(self):return self._tok
