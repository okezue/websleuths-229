from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from wm.core.schema import EvidenceSpan
from wm.evidence.store import EvidenceStore


@dataclass
class SearchHit:
    span: EvidenceSpan
    score: float
    lexical_score: float = 0.0
    semantic_score: float = 0.0


class HybridIndex:
    """FTS + local TF-IDF index. It is deliberately inspectable and deterministic."""

    def __init__(self, store: EvidenceStore):
        self.store = store
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        self.span_ids: list[str] = []

    def rebuild(self, spans: Iterable[EvidenceSpan] | None = None) -> None:
        items = list(spans if spans is not None else self.store.get_spans())
        texts = [span.text for span in items]
        self.span_ids = [span.span_id for span in items]
        if not texts:
            self.vectorizer = None
            self.matrix = None
            return
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_features=100_000,
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        modalities: set[str] | None = None,
        document_ids: set[str] | None = None,
        at_time: datetime | None = None,
    ) -> list[SearchHit]:
        lexical = {span.span_id: score for span, score in self.store.search_spans(query, limit=limit * 4)}
        semantic: dict[str, float] = {}
        if self.vectorizer is not None and self.matrix is not None and self.span_ids:
            q = self.vectorizer.transform([query])
            sims = cosine_similarity(q, self.matrix).ravel()
            top = np.argsort(-sims)[: limit * 4]
            semantic = {self.span_ids[int(i)]: float(sims[int(i)]) for i in top if sims[int(i)] > 0}
        ids = set(lexical) | set(semantic)
        hits: list[SearchHit] = []
        max_lex = max(lexical.values(), default=1.0) or 1.0
        for span_id in ids:
            span = self.store.get_span(span_id)
            if not span:
                continue
            if modalities and span.modality not in modalities:
                continue
            if document_ids and span.doc_id not in document_ids:
                continue
            lscore = lexical.get(span_id, 0.0) / max_lex
            sscore = semantic.get(span_id, 0.0)
            score = 0.45 * lscore + 0.55 * sscore
            hits.append(SearchHit(span=span, score=score, lexical_score=lscore, semantic_score=sscore))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:limit]
