from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn


def layer_key(name: str) -> str:
    return name.replace(".", "__").replace("/", "_")


class ResidualAdapter(nn.Module):
    """Newly allocated bottleneck neurons with a zero-effect initialization."""

    def __init__(self, hidden_size: int, rank: int, dropout: float = 0.0, scale: float = 1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank
        self.scale = scale
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # Keep the small cell in stable fp32 even when the frozen backbone runs in bf16/fp16.
        # Cast only across the residual boundary and return the backbone's original dtype.
        cell_hidden = hidden.to(dtype=self.down.weight.dtype)
        residual = self.scale * self.up(self.dropout(self.activation(self.down(cell_hidden))))
        return residual.to(dtype=hidden.dtype)


class KnowledgeCell(nn.Module):
    def __init__(
        self,
        cell_id: str,
        target_layers: Iterable[str],
        hidden_size: int,
        rank: int,
        *,
        dropout: float = 0.0,
        scale: float = 1.0,
    ):
        super().__init__()
        self.cell_id = cell_id
        self.target_layers = list(target_layers)
        self.hidden_size = hidden_size
        self.rank = rank
        self.layers = nn.ModuleDict(
            {
                layer_key(name): ResidualAdapter(hidden_size, rank, dropout=dropout, scale=scale)
                for name in self.target_layers
            }
        )

    def applies_to(self, module_name: str) -> bool:
        return layer_key(module_name) in self.layers

    def residual(self, module_name: str, hidden: torch.Tensor) -> torch.Tensor:
        return self.layers[layer_key(module_name)](hidden)

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def unfreeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        self.train()

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
