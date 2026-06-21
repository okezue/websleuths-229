from __future__ import annotations

from datetime import datetime

from wm.core.schema import ClaimRecord


def supersede(old: ClaimRecord, new: ClaimRecord, effective_at: datetime) -> tuple[ClaimRecord, ClaimRecord]:
    if old.subject.lower() != new.subject.lower() or old.predicate.lower() != new.predicate.lower():
        raise ValueError("only claims with the same subject and predicate can supersede one another")
    closed = old.model_copy(update={"valid_to": effective_at, "status": "superseded"})
    opened = new.model_copy(update={"valid_from": effective_at})
    return closed, opened
