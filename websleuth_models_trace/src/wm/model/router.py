from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from wm.core.io import read_json, write_json
from wm.core.schema import CellMetadata, RouteDecision
from wm.core.time import active_at

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


class HardRouter:
    def __init__(
        self,
        *,
        threshold: float = 0.22,
        max_active: int = 2,
        entity_bonus: float = 0.35,
    ):
        self.threshold = threshold
        self.max_active = max_active
        self.entity_bonus = entity_bonus
        self.metadata: dict[str, CellMetadata] = {}
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        self.ids: list[str] = []

    def add(self, metadata: CellMetadata) -> None:
        self.metadata[metadata.cell_id] = metadata
        self.rebuild()

    def remove(self, cell_id: str) -> None:
        self.metadata.pop(cell_id, None)
        self.rebuild()

    def rebuild(self) -> None:
        self.ids = sorted(self.metadata)
        if not self.ids:
            self.vectorizer = None
            self.matrix = None
            return
        texts = [self.metadata[cell_id].route_text + " " + " ".join(self.metadata[cell_id].entities) for cell_id in self.ids]
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(texts)

    @staticmethod
    def _entities_in_query(query: str, entities: Iterable[str]) -> int:
        q = query.lower()
        return sum(1 for entity in entities if entity and entity.lower() in q)

    def select(self, query: str, at_time: datetime | None = None) -> RouteDecision:
        when = at_time or datetime.now(UTC)
        if not self.ids or self.vectorizer is None or self.matrix is None:
            return RouteDecision(query=query, reason="empty cell bank", at_time=when)
        q = self.vectorizer.transform([query])
        similarities = cosine_similarity(q, self.matrix).ravel()
        scores: dict[str, float] = {}
        for index, cell_id in enumerate(self.ids):
            meta = self.metadata[cell_id]
            if not active_at(meta.valid_from, meta.valid_to, when):
                continue
            score = float(similarities[index])
            matches = self._entities_in_query(query, meta.entities)
            if matches:
                score += self.entity_bonus * min(matches, 2)
            scores[cell_id] = score
        selected = [cell_id for cell_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True) if score >= self.threshold][
            : self.max_active
        ]
        reason = "routed" if selected else "all scores below threshold"
        return RouteDecision(query=query, cell_ids=selected, scores=scores, reason=reason, at_time=when)

    def false_positive_rate(self, prompts: Iterable[str], expected_empty: bool = True) -> float:
        prompts = list(prompts)
        if not prompts:
            return 0.0
        positives = sum(bool(self.select(prompt).cell_ids) for prompt in prompts)
        return positives / len(prompts)

    def save(self, path: str | Path) -> None:
        write_json(
            path,
            {
                "threshold": self.threshold,
                "max_active": self.max_active,
                "entity_bonus": self.entity_bonus,
                "cells": {cell_id: meta.model_dump(mode="json") for cell_id, meta in self.metadata.items()},
            },
        )

    @classmethod
    def load(cls, path: str | Path) -> "HardRouter":
        data = read_json(path)
        router = cls(
            threshold=float(data["threshold"]),
            max_active=int(data["max_active"]),
            entity_bonus=float(data["entity_bonus"]),
        )
        router.metadata = {cell_id: CellMetadata.model_validate(meta) for cell_id, meta in data.get("cells", {}).items()}
        router.rebuild()
        return router
