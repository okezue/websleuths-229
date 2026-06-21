from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable

from wm.core.hashing import stable_hash
from wm.core.schema import ClaimRecord, QAItem


class QuestionBuilder:
    """Builds disjoint template families so held-out scores are not same-sentence cloze tests."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    @staticmethod
    def _split(kind: str, claim_id: str) -> str:
        if kind in {"direct", "cloze"}:
            return "train"
        if kind in {"boolean"}:
            return "dev"
        return "test"

    def _item(self, claim: ClaimRecord, prompt: str, answer: str, kind: str) -> QAItem:
        return QAItem.build(
            prompt=prompt,
            answer=answer,
            kind=kind,
            split=self._split(kind, claim.claim_id),
            claim_ids=[claim.claim_id],
            proof_span_ids=claim.support_span_ids,
            metadata={"source_text": claim.text},
        )

    def build(self, claims: Iterable[ClaimRecord]) -> list[QAItem]:
        usable = [c for c in claims if c.status == "supported" and c.support_span_ids]
        items: list[QAItem] = []
        for claim in usable:
            items.extend(
                [
                    self._item(
                        claim,
                        f"What value or entity completes this verified relation: {claim.subject} — {claim.predicate}?",
                        claim.object,
                        "direct",
                    ),
                    self._item(
                        claim,
                        f"Complete the statement: {claim.subject} {claim.predicate} ____.",
                        claim.object,
                        "cloze",
                    ),
                    self._item(
                        claim,
                        f"Is the following supported by the collected evidence? {claim.text}",
                        "Yes",
                        "boolean",
                    ),
                    self._item(
                        claim,
                        f"Which subject is linked by the relation '{claim.predicate}' to '{claim.object}'?",
                        claim.subject,
                        "inverse",
                    ),
                ]
            )
            if claim.valid_from or claim.valid_to:
                interval = f"from {claim.valid_from or 'the beginning'} until {claim.valid_to or 'the present'}"
                items.append(
                    self._item(
                        claim,
                        f"During what interval is this relation valid: {claim.subject} {claim.predicate} {claim.object}?",
                        interval,
                        "temporal",
                    )
                )
        rng = random.Random(self.seed)
        by_predicate: dict[str, list[ClaimRecord]] = defaultdict(list)
        for claim in usable:
            by_predicate[claim.predicate.lower()].append(claim)
        for group in by_predicate.values():
            if len(group) < 2:
                continue
            shuffled = list(group)
            rng.shuffle(shuffled)
            for left, right in zip(shuffled[::2], shuffled[1::2]):
                false_text = f"{left.subject} {left.predicate} {right.object}."
                items.append(
                    QAItem.build(
                        prompt=f"Is the following supported by the collected evidence? {false_text}",
                        answer="No",
                        kind="negative_control",
                        split="test",
                        claim_ids=[left.claim_id, right.claim_id],
                        proof_span_ids=list(dict.fromkeys(left.support_span_ids + right.support_span_ids)),
                    )
                )
        by_subject: dict[str, list[ClaimRecord]] = defaultdict(list)
        for claim in usable:
            by_subject[claim.subject.lower()].append(claim)
        for claim in usable:
            middle = claim.object.lower()
            for next_claim in by_subject.get(middle, []):
                prompt = (
                    f"Using two verified relations, {claim.subject} {claim.predicate} {claim.object}, and "
                    f"{next_claim.subject} {next_claim.predicate} what?"
                )
                items.append(
                    QAItem.build(
                        prompt=prompt,
                        answer=next_claim.object,
                        kind="composition",
                        split="test",
                        claim_ids=[claim.claim_id, next_claim.claim_id],
                        proof_span_ids=list(dict.fromkeys(claim.support_span_ids + next_claim.support_span_ids)),
                    )
                )
        # Ensure every split exists for tiny episodes.
        existing = {item.split for item in items}
        if usable and "dev" not in existing:
            c = usable[0]
            items.append(self._item(c, f"Is '{c.object}' the supported object for {c.subject} {c.predicate}?", "Yes", "boolean"))
        if usable and "test" not in existing:
            c = usable[0]
            items.append(self._item(c, f"Identify the subject whose {c.predicate} is {c.object}.", c.subject, "inverse"))
        dedup: dict[str, QAItem] = {item.qid: item for item in items}
        return sorted(dedup.values(), key=lambda q: (q.split, q.kind, stable_hash(q.prompt)))
