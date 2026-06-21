from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

import torch

from wm.core.schema import QAItem
from wm.model.cells import KnowledgeCell
from wm.model.wrapper import TraceModel
from wm.train.losses import js_divergence


@dataclass
class EATRDState:
    lambda_value: float
    steps: int
    mean_episode_loss: float
    mean_replay_loss: float


class EATRDBaseline:
    """Shared-cell evidence adaptation with an adaptive replay-distillation constraint."""

    def __init__(
        self,
        model: TraceModel,
        rank: int = 16,
        lr: float = 1e-3,
        lambda_init: float = 0.1,
        drift_target: float = 0.2,
        seed: int = 42,
    ):
        self.model = model
        self.cell_id = "eatrd_shared"
        self.lr = lr
        self.lambda_value = lambda_init
        self.drift_target = drift_target
        self.rng = random.Random(seed)
        self.replay_prompts: list[str] = []
        self.teacher_logits: dict[str, torch.Tensor] = {}
        if self.cell_id not in model.cells:
            model.add_cell(KnowledgeCell(self.cell_id, model.target_module_names, model.hidden_size, rank), trainable=True)

    def _snapshot_replay(self) -> None:
        for prompt in self.replay_prompts:
            self.teacher_logits[prompt] = self.model.next_token_logits(prompt, cell_ids=[self.cell_id]).detach().cpu()

    def learn(self, qas: Iterable[QAItem], steps: int = 100) -> EATRDState:
        qas = list(qas)
        if not qas:
            return EATRDState(self.lambda_value, 0, 0.0, 0.0)
        self._snapshot_replay()
        cell = self.model.cells[self.cell_id]
        cell.unfreeze()
        optimizer = torch.optim.AdamW(cell.parameters(), lr=self.lr)
        ep_total = replay_total = 0.0
        for step in range(steps):
            qa = qas[step % len(qas)]
            prompt = f"Question: {qa.prompt}\nAnswer: "
            episode_loss = self.model.answer_loss(prompt, qa.answer, cell_ids=[self.cell_id])
            replay_loss = torch.tensor(0.0, device=self.model.device)
            if self.replay_prompts:
                replay_prompt = self.rng.choice(self.replay_prompts)
                current = self.model.next_token_logits(replay_prompt, cell_ids=[self.cell_id])
                teacher = self.teacher_logits[replay_prompt].to(current.device)
                replay_loss = js_divergence(current.unsqueeze(1), teacher.unsqueeze(1))
            loss = episode_loss + self.lambda_value * replay_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            drift = float(replay_loss.detach().cpu())
            if drift > self.drift_target * 1.5:
                self.lambda_value = min(10.0, self.lambda_value * 1.5)
            elif drift < self.drift_target / 1.5:
                self.lambda_value = max(0.001, self.lambda_value * 0.9)
            ep_total += float(episode_loss.detach().cpu())
            replay_total += drift
        self.replay_prompts.extend(f"Question: {qa.prompt}\nAnswer: " for qa in qas)
        self.replay_prompts = list(dict.fromkeys(self.replay_prompts))[-512:]
        return EATRDState(self.lambda_value, steps, ep_total / steps, replay_total / steps)
