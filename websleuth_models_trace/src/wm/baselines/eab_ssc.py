from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from wm.core.hashing import stable_hash
from wm.core.schema import QAItem
from wm.model.cells import KnowledgeCell, layer_key
from wm.model.wrapper import TraceModel


@dataclass
class ConsolidationState:
    episode_cells: int
    consolidated_rank: int
    approximation_error: float


class EABSSCBaseline:
    """Episode adapters followed by linearized SVD sleep consolidation."""

    def __init__(self, model: TraceModel, episode_rank: int = 8, consolidated_rank: int = 16, lr: float = 1e-3):
        self.model = model
        self.episode_rank = episode_rank
        self.consolidated_rank = consolidated_rank
        self.lr = lr
        self.episode_ids: list[str] = []
        self.consolidated_id = "eab_ssc_consolidated"

    def learn_episode(self, qas: Iterable[QAItem], steps: int = 100) -> str:
        qas = list(qas)
        cell_id = "episode_" + stable_hash([qa.qid for qa in qas], 12)
        cell = KnowledgeCell(cell_id, self.model.target_module_names, self.model.hidden_size, self.episode_rank)
        self.model.add_cell(cell, trainable=True)
        optimizer = torch.optim.AdamW(cell.parameters(), lr=self.lr)
        for step in range(steps):
            qa = qas[step % len(qas)]
            loss = self.model.answer_loss(f"Question: {qa.prompt}\nAnswer: ", qa.answer, cell_ids=[cell_id])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        cell.freeze()
        self.episode_ids.append(cell_id)
        return cell_id

    def consolidate(self) -> ConsolidationState:
        if not self.episode_ids:
            return ConsolidationState(0, self.consolidated_rank, 0.0)
        if self.consolidated_id in self.model.cells:
            self.model.remove_cell(self.consolidated_id)
        target_rank = min(self.consolidated_rank, self.model.hidden_size)
        merged = KnowledgeCell(self.consolidated_id, self.model.target_module_names, self.model.hidden_size, target_rank)
        errors: list[float] = []
        with torch.no_grad():
            for module_name in self.model.target_module_names:
                deltas = []
                for cell_id in self.episode_ids:
                    adapter = self.model.cells[cell_id].layers[layer_key(module_name)]
                    deltas.append(adapter.up.weight.float() @ adapter.down.weight.float())
                delta = torch.stack(deltas).mean(0)
                u, s, vh = torch.linalg.svd(delta, full_matrices=False)
                r = min(target_rank, s.numel())
                root = s[:r].clamp_min(0).sqrt()
                adapter = merged.layers[layer_key(module_name)]
                adapter.up.weight.zero_()
                adapter.down.weight.zero_()
                adapter.up.weight[:, :r].copy_((u[:, :r] * root).to(adapter.up.weight.dtype))
                adapter.down.weight[:r, :].copy_((root[:, None] * vh[:r, :]).to(adapter.down.weight.dtype))
                reconstruction = adapter.up.weight.float() @ adapter.down.weight.float()
                errors.append(float((delta - reconstruction).norm() / (delta.norm() + 1e-12)))
        self.model.add_cell(merged, trainable=False)
        return ConsolidationState(len(self.episode_ids), target_rank, sum(errors) / max(len(errors), 1))
