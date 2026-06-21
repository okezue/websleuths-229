from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from wm.core.schema import QAItem
from wm.evidence.store import EvidenceStore


@dataclass
class ProvenanceMetrics:
    proof_coverage: float
    valid_span_rate: float
    independent_source_rate: float
    n: int


class ProvenanceEvaluator:
    def __init__(self, store: EvidenceStore):
        self.store = store

    def evaluate(self, qas: Iterable[QAItem], min_sources: int = 2) -> ProvenanceMetrics:
        items = list(qas)
        if not items:
            return ProvenanceMetrics(0.0, 0.0, 0.0, 0)
        covered = valid = independent = 0
        for qa in items:
            spans = [span for sid in qa.proof_span_ids if (span := self.store.get_span(sid))]
            docs = [self.store.get_document(span.doc_id) for span in spans]
            if qa.claim_ids and spans:
                covered += 1
            if len(spans) == len(qa.proof_span_ids):
                valid += 1
            domains = {doc.canonical_url.split("/")[2] if "://" in doc.canonical_url else "local" for doc in docs if doc}
            if len(domains) >= min_sources:
                independent += 1
        n = len(items)
        return ProvenanceMetrics(covered / n, valid / n, independent / n, n)
