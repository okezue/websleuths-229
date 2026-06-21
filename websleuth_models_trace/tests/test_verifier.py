from __future__ import annotations

from datetime import UTC, datetime

from wm.config import ExtractionConfig
from wm.core.schema import ClaimRecord
from wm.evidence.verifier import EvidenceVerifier


def test_unsupported_candidate_cannot_contest_supported_claim() -> None:
    verifier = EvidenceVerifier(ExtractionConfig())
    supported = ClaimRecord.build(
        subject="Entity",
        predicate="has value",
        object="A",
        support_span_ids=["s1"],
        status="supported",
    )
    candidate = ClaimRecord.build(
        subject="Entity",
        predicate="has value",
        object="B",
        support_span_ids=["s2"],
        status="candidate",
    )
    left, right = verifier.mark_contradictions([supported, candidate])
    assert left.contradicts == [right.claim_id]
    assert right.contradicts == [left.claim_id]
    assert left.status == "supported"
    assert right.status == "candidate"


def test_non_overlapping_temporal_claims_are_versions_not_contradictions() -> None:
    verifier = EvidenceVerifier(ExtractionConfig())
    old = ClaimRecord.build(
        subject="Office",
        predicate="is held by",
        object="A",
        status="supported",
        valid_from=datetime(2030, 1, 1, tzinfo=UTC),
        valid_to=datetime(2030, 12, 31, tzinfo=UTC),
    )
    new = ClaimRecord.build(
        subject="Office",
        predicate="is held by",
        object="B",
        status="supported",
        valid_from=datetime(2031, 1, 1, tzinfo=UTC),
    )
    old_out, new_out = verifier.mark_contradictions([old, new])
    assert not old_out.contradicts
    assert not new_out.contradicts
    assert old_out.status == new_out.status == "supported"
