import pytest
import tempfile,os
from wm.types import Episode
from wm.ingest.stub import StubSrc
from wm.cfg import WMCfg,ChunkCfg
from datetime import datetime
from transformers import AutoTokenizer,AutoModelForCausalLM,LlamaConfig
from peft import get_peft_model,LoraConfig,TaskType

@pytest.fixture
def stub_eps():
    return StubSrc(3).fetch("test")

@pytest.fixture
def single_ep():
    return Episode(
        url="https://example.com/article-1",
        title="Test Article",
        body="This is a test article body with enough content. "*20,
        ts=datetime(2025,6,1),
    )

@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d

@pytest.fixture
def wm_cfg():
    return WMCfg()

@pytest.fixture
def chunk_cfg():
    return ChunkCfg(max_tok=64,overlap=8)

@pytest.fixture(scope="session")
def tiny_model():
    cfg=LlamaConfig(
        vocab_size=256,hidden_size=64,intermediate_size=128,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,
        max_position_embeddings=512,
    )
    m=AutoModelForCausalLM.from_config(cfg)
    return m

@pytest.fixture(scope="session")
def tiny_tok(tmp_path_factory):
    from transformers import PreTrainedTokenizerFast
    from tokenizers import Tokenizer,models,trainers,pre_tokenizers
    tok=Tokenizer(models.BPE())
    tok.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
    tr=trainers.BpeTrainer(vocab_size=256,special_tokens=["<pad>","<s>","</s>"])
    tok.train_from_iterator(["hello world this is test data "*100],trainer=tr)
    d=str(tmp_path_factory.mktemp("tok"))
    tok.save(os.path.join(d,"tokenizer.json"))
    ft=PreTrainedTokenizerFast(tokenizer_file=os.path.join(d,"tokenizer.json"))
    ft.pad_token="<pad>"
    ft.bos_token="<s>"
    ft.eos_token="</s>"
    return ft
