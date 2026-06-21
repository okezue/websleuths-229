from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

import torch

from wm.core.schema import QAItem
from wm.model.cells import KnowledgeCell
from wm.model.wrapper import TraceModel
from wm.train.losses import js_divergence


def _grad_vector(parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    parts = [parameter.grad.detach().reshape(-1) for parameter in parameters if parameter.grad is not None]
    return torch.cat(parts) if parts else torch.zeros(1, device=parameters[0].device)


def _set_grad_vector(parameters: list[torch.nn.Parameter], vector: torch.Tensor) -> None:
    offset = 0
    for parameter in parameters:
        size = parameter.numel()
        parameter.grad = vector[offset : offset + size].view_as(parameter).clone()
        offset += size


def _project(episode: torch.Tensor, retention: torch.Tensor) -> torch.Tensor:
    dot = torch.dot(episode, retention)
    if dot >= 0:
        return episode
    return episode - dot / (retention.norm().pow(2) + 1e-12) * retention


@dataclass
class DPMUState:
    steps: int
    projected_fraction: float
    mean_loss: float


class DPMUBaseline:
    """Shared-cell gradient projection against cached old-behavior constraints."""

    def __init__(self, model: TraceModel, rank: int = 16, lr: float = 1e-3, seed: int = 42):
        self.model = model
        self.cell_id = "dpmu_shared"
        self.lr = lr
        self.rng = random.Random(seed)
        self.replay_prompts: list[str] = []
        self.teacher_logits: dict[str, torch.Tensor] = {}
        if self.cell_id not in model.cells:
            model.add_cell(KnowledgeCell(self.cell_id, model.target_module_names, model.hidden_size, rank), trainable=True)

    def learn(self, qas: Iterable[QAItem], steps: int = 100) -> DPMUState:
        qas = list(qas)
        if not qas:
            return DPMUState(0, 0.0, 0.0)
        for prompt in self.replay_prompts:
            self.teacher_logits[prompt] = self.model.next_token_logits(prompt, cell_ids=[self.cell_id]).detach().cpu()
        cell = self.model.cells[self.cell_id]
        cell.unfreeze()
        parameters = [p for p in cell.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=self.lr)
        projected = 0
        total_loss = 0.0
        for step in range(steps):
            qa = qas[step % len(qas)]
            prompt = f"Question: {qa.prompt}\nAnswer: "
            optimizer.zero_grad(set_to_none=True)
            loss = self.model.answer_loss(prompt, qa.answer, cell_ids=[self.cell_id])
            loss.backward()
            episode_grad = _grad_vector(parameters).clone()
            final_grad = episode_grad
            if self.replay_prompts:
                replay_prompt = self.rng.choice(self.replay_prompts)
                optimizer.zero_grad(set_to_none=True)
                current = self.model.next_token_logits(replay_prompt, cell_ids=[self.cell_id])
                teacher = self.teacher_logits[replay_prompt].to(current.device)
                retention_loss = js_divergence(current.unsqueeze(1), teacher.unsqueeze(1))
                retention_loss.backward()
                retention_grad = _grad_vector(parameters).clone()
                if torch.dot(episode_grad, retention_grad) < 0:
                    projected += 1
                final_grad = _project(episode_grad, retention_grad)
            optimizer.zero_grad(set_to_none=True)
            _set_grad_vector(parameters, final_grad)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
        self.replay_prompts.extend(f"Question: {qa.prompt}\nAnswer: " for qa in qas)
        self.replay_prompts = list(dict.fromkeys(self.replay_prompts))[-512:]
        return DPMUState(steps, projected / steps, total_loss / steps)
