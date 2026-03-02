from __future__ import annotations
from wm.cfg import TrainCfg

def make_back(cfg:TrainCfg):
    if cfg.backend=="hf":
        from wm.train.hf import HFBack
        return HFBack(cfg)
    if cfg.backend=="neuron":
        from wm.train.neuron import NeuronBack
        return NeuronBack(cfg)
    if cfg.backend=="tinker":
        from wm.train.tinker import TinkerBack
        return TinkerBack(cfg)
    raise ValueError(f"unknown backend: {cfg.backend}")
