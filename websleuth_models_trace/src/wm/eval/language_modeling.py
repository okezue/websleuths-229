from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F

from wm.model.router import HardRouter
from wm.model.wrapper import TraceModel


@dataclass
class LMPoint:
    context_length: int
    nll: float
    perplexity: float
    prefill_seconds_per_1k_tokens: float
    tokens: int


class LongContextLanguageModelingRunner:
    """Measures loss and prefill latency as context grows, including token-position curves."""

    def __init__(self, model: TraceModel, router: HardRouter | None = None):
        self.model = model
        self.router = router

    def _tokens(self, text: str, max_length: int) -> dict[str, torch.Tensor]:
        encoded = self.model.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        return {key: value.to(self.model.device) for key, value in encoded.items() if isinstance(value, torch.Tensor)}

    def evaluate_text(self, text: str, context_lengths: Iterable[int]) -> list[LMPoint]:
        points: list[LMPoint] = []
        route = self.router.select(text[:1000]).cell_ids if self.router else []
        for length in context_lengths:
            encoded = self._tokens(text, length)
            tokens = int(encoded["attention_mask"].sum())
            start = time.perf_counter()
            with self.model.activated(route), torch.no_grad():
                output = self.model.base_model(**encoded, labels=encoded["input_ids"])
            elapsed = time.perf_counter() - start
            nll = float(output.loss.detach().cpu())
            points.append(
                LMPoint(
                    context_length=length,
                    nll=nll,
                    perplexity=math.exp(min(nll, 20)),
                    prefill_seconds_per_1k_tokens=elapsed / max(tokens, 1) * 1000,
                    tokens=tokens,
                )
            )
        return points

    def token_position_loss(self, text: str, context_length: int, bins: int = 32) -> list[dict[str, float]]:
        encoded = self._tokens(text, context_length)
        route = self.router.select(text[:1000]).cell_ids if self.router else []
        with self.model.activated(route), torch.no_grad():
            logits = self.model.base_model(**encoded).logits[:, :-1, :]
        targets = encoded["input_ids"][:, 1:]
        losses = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none").squeeze(0)
        width = max(1, math.ceil(losses.numel() / bins))
        curve = []
        for start in range(0, losses.numel(), width):
            segment = losses[start : start + width]
            curve.append(
                {
                    "start_token": float(start + 1),
                    "end_token": float(start + segment.numel()),
                    "mean_nll": float(segment.mean().cpu()),
                }
            )
        return curve
