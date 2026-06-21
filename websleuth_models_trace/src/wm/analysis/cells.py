from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import torch

from wm.model.cells import layer_key
from wm.model.wrapper import TraceModel


@dataclass
class ActivationRecord:
    cell_id: str
    layer: str
    mean_norm: float
    max_norm: float
    calls: int


class CellActivationTracer:
    """Measures whether newly allocated neurons are actually used on routed prompts."""

    def __init__(self, model: TraceModel):
        self.model = model

    def trace(self, prompts: Iterable[str], routes: dict[str, list[str]]) -> list[ActivationRecord]:
        values: dict[tuple[str, str], list[float]] = defaultdict(list)
        handles = []
        for cell_id, cell in self.model.cells.items():
            for module_name in cell.target_layers:
                adapter = cell.layers[layer_key(module_name)]

                def hook(_module, _inputs, output, cid=cell_id, layer=module_name):
                    if isinstance(output, torch.Tensor):
                        values[(cid, layer)].append(float(output.detach().float().norm(dim=-1).mean().cpu()))

                handles.append(adapter.register_forward_hook(hook))
        try:
            for prompt in prompts:
                cell_ids = routes.get(prompt, [])
                self.model.next_token_logits(prompt, cell_ids=cell_ids)
        finally:
            for handle in handles:
                handle.remove()
        records = []
        for (cell_id, layer), norms in sorted(values.items()):
            records.append(ActivationRecord(cell_id, layer, sum(norms) / len(norms), max(norms), len(norms)))
        return records
