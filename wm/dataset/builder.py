from __future__ import annotations
from datasets import Dataset
from wm.types import Chunk
from wm.cfg import DatasetCfg
from wm.dataset.cpt import CptRecipe
from wm.dataset.ext_sft import ExtSftRecipe
from wm.dataset.cited_qa import CitedQaRecipe

_RECIPES={"cpt":CptRecipe,"ext_sft":ExtSftRecipe,"cited_qa":CitedQaRecipe}

class DatasetBuilder:
    def __init__(self,cfg:DatasetCfg):
        self._cfg=cfg
        self._recipe=_RECIPES[cfg.recipe](max_len=cfg.max_len)
    def build(self,chunks:list[Chunk])->Dataset:
        rows=self._recipe.build(chunks)
        if not rows:
            return Dataset.from_dict({"text":[]})
        return Dataset.from_list(rows)
    def build_split(self,chunks:list[Chunk])->tuple[Dataset,Dataset]:
        ds=self.build(chunks)
        if len(ds)<2:
            return ds,ds
        sp=ds.train_test_split(test_size=self._cfg.test_frac,seed=self._cfg.seed)
        return sp["train"],sp["test"]
