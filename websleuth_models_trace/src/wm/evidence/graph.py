from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import networkx as nx

from wm.core.schema import ClaimRecord
from wm.core.time import active_at


@dataclass
class ProofAnswer:
    answer: str
    claim_ids: list[str]
    proof_span_ids: list[str]
    confidence: float
    explanation: str


class EvidenceGraph:
    def __init__(self, claims: Iterable[ClaimRecord] = ()): 
        self.graph = nx.MultiDiGraph()
        self.claims: dict[str, ClaimRecord] = {}
        for claim in claims:
            self.add_claim(claim)

    def add_claim(self, claim: ClaimRecord) -> None:
        self.claims[claim.claim_id] = claim
        self.graph.add_node(claim.subject, kind="entity")
        self.graph.add_node(claim.object, kind="value")
        self.graph.add_edge(
            claim.subject,
            claim.object,
            key=claim.claim_id,
            relation=claim.predicate,
            claim_id=claim.claim_id,
            confidence=claim.confidence,
        )

    def active_claims(self, when: datetime | None = None) -> list[ClaimRecord]:
        if when is None:
            return [c for c in self.claims.values() if c.status in {"supported", "candidate", "contested"}]
        return [
            c
            for c in self.claims.values()
            if c.status in {"supported", "candidate", "contested"} and active_at(c.valid_from, c.valid_to, when)
        ]

    def find(self, subject: str | None = None, predicate: str | None = None, obj: str | None = None, when: datetime | None = None) -> list[ClaimRecord]:
        def match(needle: str | None, value: str) -> bool:
            return needle is None or needle.lower() in value.lower()

        return [
            claim
            for claim in self.active_claims(when)
            if match(subject, claim.subject) and match(predicate, claim.predicate) and match(obj, claim.object)
        ]

    def answer(self, question: str, when: datetime | None = None) -> ProofAnswer | None:
        q = question.lower()
        candidates = self.active_claims(when)
        scored: list[tuple[float, ClaimRecord]] = []
        q_tokens = set(q.replace("?", "").split())
        for claim in candidates:
            text = f"{claim.subject} {claim.predicate} {claim.object}".lower()
            overlap = len(q_tokens & set(text.split())) / max(len(q_tokens), 1)
            if claim.subject.lower() in q:
                overlap += 0.35
            if claim.predicate.lower() in q:
                overlap += 0.25
            scored.append((overlap * claim.confidence, claim))
        if not scored:
            return None
        score, claim = max(scored, key=lambda item: item[0])
        if score <= 0:
            return None
        if q.startswith(("which", "who")) and claim.object.lower() in q:
            answer = claim.subject
        else:
            answer = claim.object
        return ProofAnswer(
            answer=answer,
            claim_ids=[claim.claim_id],
            proof_span_ids=claim.support_span_ids,
            confidence=claim.confidence,
            explanation=claim.text,
        )

    def connected_components(self) -> list[list[str]]:
        undirected = self.graph.to_undirected()
        return [sorted(component) for component in nx.connected_components(undirected)]
