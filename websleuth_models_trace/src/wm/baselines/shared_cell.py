from __future__ import annotations

import random
from collections import deque
from typing import Iterable

import torch

from wm.core.schema import QAItem
from wm.model.cells import KnowledgeCell
from wm.model.wrapper import TraceModel


class SharedCellBaseline:
    """One shared mutable cell: a direct catastrophic-forgetting baseline."""

    def __init__(self, model: TraceModel, rank: int = 16, lr: float = 1e-3):
        self.model = model
        self.cell_id = "shared_mutable_cell"
        self.lr = lr
        if self.cell_id not in model.cells:
            model.add_cell(
                KnowledgeCell(self.cell_id, model.target_module_names, model.hidden_size, rank),
                trainable=True,
            )
        self.optimizer = torch.optim.AdamW(model.cells[self.cell_id].parameters(), lr=lr)

    def learn(self, qas: Iterable[QAItem], steps: int = 100, replay: Iterable[QAItem] = ()) -> list[float]:
        examples = list(qas) + list(replay)
        if not examples:
            return []
        history: list[float] = []
        for step in range(steps):
            qa = examples[step % len(examples)]
            prompt = f"Question: {qa.prompt}\nAnswer: "
            loss = self.model.answer_loss(prompt, qa.answer, cell_ids=[self.cell_id])
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            history.append(float(loss.detach().cpu()))
        return history

    def answer(self, prompt: str) -> str:
        return self.model.generate_text(prompt, cell_ids=[self.cell_id])


class ReplaySharedCellBaseline(SharedCellBaseline):
    def __init__(self, model: TraceModel, rank: int = 16, lr: float = 1e-3, replay_size: int = 256, seed: int = 42):
        super().__init__(model, rank=rank, lr=lr)
        self.buffer: deque[QAItem] = deque(maxlen=replay_size)
        self.rng = random.Random(seed)

    def learn_episode(self, qas: Iterable[QAItem], steps: int = 100, replay_ratio: float = 0.5) -> list[float]:
        current = list(qas)
        replay_n = min(len(self.buffer), int(max(0.0, replay_ratio) * max(len(current), 1)))
        replay = self.rng.sample(list(self.buffer), replay_n) if replay_n else []
        history = self.learn(current, steps=steps, replay=replay)
        self.buffer.extend(current)
        return history
