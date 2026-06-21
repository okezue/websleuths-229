from __future__ import annotations

import math
import uuid
from typing import Iterable, Sequence

import numpy as np
import torch

from wm.model.cells import KnowledgeCell
from wm.model.wrapper import TraceModel


def _output_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    return None


class GradientSpectrumRankEstimator:
    """Estimate cell rank from held-out residual gradient directions.

    A zero-effect rank-1 probe cell makes target-layer outputs differentiable without
    unfreezing the backbone. For each evidence QA we collect the loss gradient with
    respect to every selected MLP output, producing a small matrix of residual directions
    in hidden space. The chosen rank is the smallest rank that explains ``energy`` of its
    singular spectrum. This measures the dimensionality of the update actually demanded
    by the evidence rather than the dimensionality of ordinary hidden activations.
    """

    def __init__(
        self,
        min_rank: int = 4,
        max_rank: int = 64,
        energy: float = 0.9,
        max_examples: int = 16,
    ):
        self.min_rank = min_rank
        self.max_rank = max_rank
        self.energy = energy
        self.max_examples = max_examples

    def estimate(
        self,
        model: TraceModel,
        examples: Iterable[tuple[str, str, Sequence[str]]],
    ) -> tuple[int, dict[str, float]]:
        examples = list(examples)[: self.max_examples]
        if not examples:
            return self.min_rank, {
                "effective_rank": float(self.min_rank),
                "captured_energy": 1.0,
                "gradient_directions": 0.0,
            }
        probe_id = "__rank_probe_" + uuid.uuid4().hex
        probe = KnowledgeCell(
            probe_id,
            model.target_module_names,
            model.hidden_size,
            rank=1,
            dropout=0.0,
            scale=1.0,
        )
        model.add_cell(probe, trainable=True)
        modules = dict(model.base_model.named_modules())
        current_outputs: list[torch.Tensor] = []
        handles = []

        def hook(_module, _inputs, output):
            tensor = _output_tensor(output)
            if tensor is not None and tensor.requires_grad:
                tensor.retain_grad()
                current_outputs.append(tensor)
            return output

        for name in model.target_module_names:
            handles.append(modules[name].register_forward_hook(hook))
        directions: list[torch.Tensor] = []
        try:
            for prompt, answer, existing_cells in examples:
                current_outputs.clear()
                model.zero_grad(set_to_none=True)
                active = list(dict.fromkeys(list(existing_cells) + [probe_id]))
                loss = model.answer_loss(prompt, answer, cell_ids=active)
                loss.backward()
                for tensor in current_outputs:
                    if tensor.grad is None:
                        continue
                    grad = tensor.grad.detach().float()
                    reduce_dims = tuple(range(grad.ndim - 1))
                    vector = grad.mean(dim=reduce_dims) if reduce_dims else grad
                    norm = vector.norm()
                    if torch.isfinite(norm) and float(norm) > 0:
                        directions.append((vector / norm).cpu())
        finally:
            for handle in handles:
                handle.remove()
            model.remove_cell(probe_id)
            model.zero_grad(set_to_none=True)
        if len(directions) < 2:
            return self.min_rank, {
                "effective_rank": float(self.min_rank),
                "captured_energy": 1.0,
                "gradient_directions": float(len(directions)),
            }
        matrix = torch.stack(directions).numpy()
        singular = np.linalg.svd(matrix, compute_uv=False, full_matrices=False)
        values = singular**2
        total = float(values.sum())
        if total <= 0:
            rank = self.min_rank
            captured = 1.0
        else:
            cumulative = np.cumsum(values) / total
            effective = int(np.searchsorted(cumulative, self.energy) + 1)
            rank = max(self.min_rank, min(self.max_rank, model.hidden_size, effective))
            captured = float(cumulative[min(rank - 1, len(cumulative) - 1)])
        return rank, {
            "effective_rank": float(rank),
            "captured_energy": captured,
            "gradient_directions": float(len(directions)),
            "spectrum_total": total,
            "top_singular": float(singular[0]) if len(singular) else 0.0,
        }


class HiddenSpectrumRankEstimator:
    """Fallback estimator based on target-layer hidden-state energy."""

    def __init__(self, min_rank: int = 4, max_rank: int = 64, energy: float = 0.9):
        self.min_rank = min_rank
        self.max_rank = max_rank
        self.energy = energy

    def estimate(self, model: TraceModel, prompts: Iterable[str]) -> tuple[int, dict[str, float]]:
        prompts = list(prompts)
        if not prompts or model.tokenizer is None:
            return self.min_rank, {"effective_rank": float(self.min_rank), "captured_energy": 1.0}
        captured: list[torch.Tensor] = []
        hooks = []
        modules = dict(model.base_model.named_modules())

        def make_hook():
            def hook(_module, _inputs, output):
                tensor = output[0] if isinstance(output, tuple) else output
                if isinstance(tensor, torch.Tensor):
                    captured.append(tensor.detach().float().mean(dim=tuple(range(tensor.ndim - 1))).cpu())
            return hook

        for name in model.target_module_names:
            hooks.append(modules[name].register_forward_hook(make_hook()))
        try:
            with model.activated([]), torch.no_grad():
                for prompt in prompts[: max(self.max_rank * 2, 16)]:
                    model.base_model(**model._encode(prompt))
        finally:
            for hook in hooks:
                hook.remove()
        if len(captured) < 2:
            return self.min_rank, {"effective_rank": float(self.min_rank), "captured_energy": 1.0}
        matrix = torch.stack(captured).numpy()
        matrix = matrix - matrix.mean(axis=0, keepdims=True)
        singular = np.linalg.svd(matrix, compute_uv=False, full_matrices=False)
        energy_values = singular**2
        total = float(energy_values.sum())
        if total <= 0:
            rank = self.min_rank
            captured_energy = 1.0
        else:
            cumulative = np.cumsum(energy_values) / total
            effective = int(np.searchsorted(cumulative, self.energy) + 1)
            rank = max(self.min_rank, min(self.max_rank, effective))
            captured_energy = float(cumulative[min(rank - 1, len(cumulative) - 1)])
        return rank, {
            "effective_rank": float(rank),
            "captured_energy": captured_energy,
            "samples": float(len(captured)),
            "spectrum_total": total,
        }


def rank_from_residual(residual: float, min_rank: int, max_rank: int) -> int:
    residual = max(0.0, residual)
    fraction = 1.0 - math.exp(-residual)
    return max(min_rank, min(max_rank, int(round(min_rank + fraction * (max_rank - min_rank)))))
