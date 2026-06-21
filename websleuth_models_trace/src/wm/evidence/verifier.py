from __future__ import annotations

import re
from collections import defaultdict
from datetime import UTC, datetime
from urllib.parse import urlparse

from wm.config import ExtractionConfig
from wm.core.schema import ClaimRecord, DocumentRecord, EvidenceSpan

_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "at", "for", "and", "or", "is", "are", "was", "were",
    "has", "had", "with", "by", "from", "that", "this", "as", "it", "its",
}
_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?%?")
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._%+-]*")


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text) if t.lower() not in _STOPWORDS}


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUMBER_RE.findall(text)}


def support_score(claim: ClaimRecord, span: EvidenceSpan) -> float:
    claim_tokens = _tokens(claim.text)
    span_tokens = _tokens(span.text)
    overlap = len(claim_tokens & span_tokens) / max(len(claim_tokens), 1)
    object_bonus = 0.25 if claim.object.lower() in span.text.lower() else 0.0
    subject_bonus = 0.15 if claim.subject.lower() in span.text.lower() else 0.0
    claim_numbers = _numbers(claim.text)
    numeric = 1.0 if not claim_numbers or claim_numbers <= _numbers(span.text) else 0.0
    return min(1.0, 0.6 * overlap + object_bonus + subject_bonus) * numeric


class EvidenceVerifier:
    def __init__(self, cfg: ExtractionConfig):
        self.cfg = cfg

    def verify(
        self,
        claims: list[ClaimRecord],
        spans: dict[str, EvidenceSpan],
        documents: dict[str, DocumentRecord],
    ) -> list[ClaimRecord]:
        verified: list[ClaimRecord] = []
        for claim in claims:
            valid_support: list[str] = []
            domains: list[str] = []
            scores: list[float] = []
            for span_id in claim.support_span_ids:
                span = spans.get(span_id)
                if not span:
                    continue
                score = support_score(claim, span)
                if score >= self.cfg.min_support_overlap:
                    valid_support.append(span_id)
                    scores.append(score)
                    doc = documents.get(span.doc_id)
                    if doc:
                        parsed = doc.metadata.get("parse", {}) if isinstance(doc.metadata, dict) else {}
                        source_label = parsed.get("source_domain", "") if isinstance(parsed, dict) else ""
                        if source_label:
                            domains.append(str(source_label))
                        else:
                            parsed_url = urlparse(doc.canonical_url)
                            if parsed_url.netloc:
                                domains.append(parsed_url.netloc)
                            else:
                                from pathlib import Path as _Path
                                domains.append(_Path(parsed_url.path).parent.name or "local")
            independent = len(set(domains))
            if not valid_support:
                status = "rejected"
                confidence = 0.0
            elif independent >= self.cfg.require_independent_sources:
                status = "supported"
                confidence = min(1.0, 0.55 + 0.1 * independent + 0.25 * sum(scores) / len(scores))
            else:
                status = "candidate"
                confidence = min(0.79, 0.4 + 0.25 * sum(scores) / len(scores))
            verified.append(
                claim.model_copy(
                    update={
                        "support_span_ids": valid_support,
                        "source_domains": sorted(set(domains)),
                        "status": status,
                        "confidence": confidence,
                    }
                )
            )
        return self.mark_contradictions(verified)

    @staticmethod
    def _intervals_overlap(left: ClaimRecord, right: ClaimRecord) -> bool:
        low = datetime.min.replace(tzinfo=UTC)
        high = datetime.max.replace(tzinfo=UTC)
        left_start = left.valid_from or low
        left_end = left.valid_to or high
        right_start = right.valid_from or low
        right_end = right.valid_to or high
        return max(left_start, right_start) <= min(left_end, right_end)

    def mark_contradictions(self, claims: list[ClaimRecord]) -> list[ClaimRecord]:
        groups: dict[tuple[str, str], list[ClaimRecord]] = defaultdict(list)
        for claim in claims:
            key = (" ".join(claim.subject.lower().split()), " ".join(claim.predicate.lower().split()))
            groups[key].append(claim)
        all_conflicts: dict[str, list[str]] = defaultdict(list)
        supported_conflicts: dict[str, list[str]] = defaultdict(list)
        for group in groups.values():
            for i, left in enumerate(group):
                for right in group[i + 1 :]:
                    objects_differ = " ".join(left.object.lower().split()) != " ".join(right.object.lower().split())
                    if not objects_differ or not self._intervals_overlap(left, right):
                        continue
                    all_conflicts[left.claim_id].append(right.claim_id)
                    all_conflicts[right.claim_id].append(left.claim_id)
                    # A low-evidence candidate cannot, by itself, make a corroborated claim
                    # unsafe for parameterization. Two independently supported alternatives
                    # are genuinely unresolved and both become contested.
                    if left.status == "supported" and right.status == "supported":
                        supported_conflicts[left.claim_id].append(right.claim_id)
                        supported_conflicts[right.claim_id].append(left.claim_id)
        result: list[ClaimRecord] = []
        for claim in claims:
            conflicts = sorted(set(all_conflicts.get(claim.claim_id, [])))
            status = "contested" if supported_conflicts.get(claim.claim_id) else claim.status
            result.append(claim.model_copy(update={"contradicts": conflicts, "status": status}))
        return result

