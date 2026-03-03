from __future__ import annotations
import torch

def expected_gain(g_k:torch.Tensor,g_a:torch.Tensor,
                  alpha:float=0.5)->float:
    norm_sq=g_k.dot(g_k).item()
    dot=g_a.dot(g_k).item()
    penalty=alpha*max(0.0,-dot)
    return norm_sq-penalty
