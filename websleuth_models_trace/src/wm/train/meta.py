from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from wm.core.schema import EvidenceEpisode, QAItem
from wm.model.cells import KnowledgeCell
from wm.model.initializer import CellInitializer
from wm.model.wrapper import TraceModel


@dataclass
class MetaHistory:
    iteration: int
    train_loss: float
    dev_loss: float


class ReptileCellMetaLearner:
    """First-order outer loop that learns an initialization for rapid evidence assimilation.

    Each evidence episode is a task. The inner loop adapts a temporary cell on train queries;
    the outer loop moves the initialization toward the adapted parameters. This avoids
    second-order memory while directly optimizing for fast post-update performance.
    """

    def __init__(
        self,
        model: TraceModel,
        rank: int,
        *,
        inner_lr: float = 1e-3,
        inner_steps: int = 8,
        meta_step_size: float = 0.1,
    ):
        self.model = model
        self.rank = rank
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.meta_step_size = meta_step_size
        self.template = KnowledgeCell(
            "meta_template",
            model.target_module_names,
            model.hidden_size,
            rank,
        ).to(model.device)

    @staticmethod
    def _qas(episode: EvidenceEpisode, split: str) -> list[QAItem]:
        return [qa for qa in episode.qa_items if qa.split == split]

    def _loss(self, cell_id: str, qa: QAItem) -> torch.Tensor:
        return self.model.answer_loss(f"Question: {qa.prompt}\nAnswer: ", qa.answer, cell_ids=[cell_id])

    def train(self, episodes: Iterable[EvidenceEpisode], iterations: int = 100) -> list[MetaHistory]:
        episodes = [episode for episode in episodes if self._qas(episode, "train") and self._qas(episode, "dev")]
        if not episodes:
            raise ValueError("meta-training requires episodes with train and dev queries")
        history: list[MetaHistory] = []
        for iteration in range(iterations):
            episode = episodes[iteration % len(episodes)]
            cell_id = f"meta_inner_{iteration}"
            inner = KnowledgeCell(cell_id, self.model.target_module_names, self.model.hidden_size, self.rank).to(self.model.device)
            inner.load_state_dict(copy.deepcopy(self.template.state_dict()))
            self.model.add_cell(inner, trainable=True)
            optimizer = torch.optim.SGD(inner.parameters(), lr=self.inner_lr)
            train_qas = self._qas(episode, "train")
            train_total = 0.0
            for step in range(self.inner_steps):
                qa = train_qas[step % len(train_qas)]
                loss = self._loss(cell_id, qa)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                train_total += float(loss.detach().cpu())
            with torch.no_grad():
                for target, adapted in zip(self.template.parameters(), inner.parameters(), strict=True):
                    target.add_(self.meta_step_size * (adapted.detach() - target))
            dev_qas = self._qas(episode, "dev")
            with torch.no_grad():
                dev_loss = sum(float(self._loss(cell_id, qa).detach().cpu()) for qa in dev_qas) / len(dev_qas)
            self.model.remove_cell(cell_id)
            history.append(MetaHistory(iteration, train_total / self.inner_steps, dev_loss))
        return history

    def save(self, path: str | Path) -> None:
        CellInitializer.save(self.template, path)
