from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from wm.core.hashing import stable_hash
from wm.core.time import utcnow


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Locator(StrictModel):
    dom_path: str | None = None
    selector: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    region: tuple[int, int, int, int] | None = None
    time_start: float | None = None
    time_end: float | None = None
    frame: int | None = None
    table_row: int | None = None
    table_col: int | None = None


class DocumentRecord(StrictModel):
    doc_id: str
    url: str
    canonical_url: str
    title: str = ""
    mime_type: str = "application/octet-stream"
    fetched_at: datetime = Field(default_factory=utcnow)
    content_hash: str
    blob_path: str
    status_code: int = 200
    parent_doc_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def build(
        cls,
        *,
        url: str,
        canonical_url: str | None,
        content_hash: str,
        blob_path: str,
        **kwargs: Any,
    ) -> "DocumentRecord":
        canonical = canonical_url or url
        return cls(
            doc_id=stable_hash({"url": canonical, "content_hash": content_hash}),
            url=url,
            canonical_url=canonical,
            content_hash=content_hash,
            blob_path=blob_path,
            **kwargs,
        )


class EvidenceSpan(StrictModel):
    span_id: str
    doc_id: str
    modality: Literal["html", "text", "pdf", "image", "video", "audio", "table", "code"]
    text: str = ""
    locator: Locator = Field(default_factory=Locator)
    content_hash: str
    parent_span_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def build(
        cls,
        *,
        doc_id: str,
        modality: str,
        text: str,
        locator: Locator | None = None,
        metadata: dict[str, Any] | None = None,
        parent_span_id: str | None = None,
    ) -> "EvidenceSpan":
        locator = locator or Locator()
        payload = {
            "doc_id": doc_id,
            "modality": modality,
            "text": text,
            "locator": locator.model_dump(mode="json"),
        }
        return cls(
            span_id=stable_hash(payload),
            doc_id=doc_id,
            modality=modality,  # type: ignore[arg-type]
            text=text,
            locator=locator,
            content_hash=stable_hash(text, length=None),
            parent_span_id=parent_span_id,
            metadata=metadata or {},
        )


class ClaimRecord(StrictModel):
    claim_id: str
    subject: str
    predicate: str
    object: str
    text: str
    support_span_ids: list[str] = Field(default_factory=list)
    contradicts: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    status: Literal["candidate", "supported", "contested", "superseded", "rejected"] = "candidate"
    source_domains: list[str] = Field(default_factory=list)
    qualifiers: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def build(
        cls,
        *,
        subject: str,
        predicate: str,
        object: str,
        text: str | None = None,
        support_span_ids: list[str] | None = None,
        **kwargs: Any,
    ) -> "ClaimRecord":
        canonical = {
            "subject": " ".join(subject.lower().split()),
            "predicate": " ".join(predicate.lower().split()),
            "object": " ".join(object.lower().split()),
        }
        return cls(
            claim_id=stable_hash(canonical),
            subject=subject.strip(),
            predicate=predicate.strip(),
            object=object.strip(),
            text=(text or f"{subject} {predicate} {object}.").strip(),
            support_span_ids=support_span_ids or [],
            **kwargs,
        )

    @field_validator("subject", "predicate", "object", "text")
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("claim fields must be non-empty")
        return value.strip()


class QAItem(StrictModel):
    qid: str
    prompt: str
    answer: str
    kind: Literal[
        "direct",
        "inverse",
        "cloze",
        "boolean",
        "temporal",
        "composition",
        "negative_control",
    ]
    split: Literal["train", "dev", "test"]
    claim_ids: list[str] = Field(default_factory=list)
    proof_span_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def build(cls, *, prompt: str, answer: str, kind: str, split: str, **kwargs: Any) -> "QAItem":
        return cls(
            qid=stable_hash({"prompt": prompt, "answer": answer, "kind": kind}),
            prompt=prompt,
            answer=answer,
            kind=kind,  # type: ignore[arg-type]
            split=split,  # type: ignore[arg-type]
            **kwargs,
        )


class EvidenceEpisode(StrictModel):
    episode_id: str
    topic: str
    domain: str
    document_ids: list[str] = Field(default_factory=list)
    span_ids: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)
    qa_items: list[QAItem] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def build(cls, *, topic: str, domain: str, claim_ids: list[str], **kwargs: Any) -> "EvidenceEpisode":
        eid = stable_hash({"topic": topic, "domain": domain, "claims": sorted(claim_ids)})
        return cls(episode_id=eid, topic=topic, domain=domain, claim_ids=claim_ids, **kwargs)


class RouteDecision(StrictModel):
    query: str
    cell_ids: list[str] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)
    reason: str = ""
    at_time: datetime | None = None


class CellMetadata(StrictModel):
    cell_id: str
    episode_id: str
    domain: str
    topic: str
    rank: int
    target_layers: list[str]
    route_text: str
    entities: list[str] = Field(default_factory=list)
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    source_hashes: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)
    proof_span_ids: list[str] = Field(default_factory=list)
    competence_probes: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    promoted: bool = False
    created_at: datetime = Field(default_factory=utcnow)
    parameter_count: int = 0
    parameter_hash: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkExample(StrictModel):
    example_id: str
    prompt: str
    answer: str
    choices: list[str] = Field(default_factory=list)
    answer_index: int | None = None
    context: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def build(cls, *, prompt: str, answer: str, **kwargs: Any) -> "BenchmarkExample":
        return cls(example_id=stable_hash({"prompt": prompt, "answer": answer}), prompt=prompt, answer=answer, **kwargs)


class BenchmarkResult(StrictModel):
    name: str
    group: str
    metrics: dict[str, float]
    n: int
    elapsed_seconds: float
    examples: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AssimilationReport(StrictModel):
    episode_id: str
    residual: dict[str, float]
    rank: int
    trained_steps: int
    accepted: bool
    reason: str
    cell_id: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    invariants: dict[str, Any] = Field(default_factory=dict)
    history: list[dict[str, float]] = Field(default_factory=list)


class StreamStepReport(StrictModel):
    step: int
    episode_id: str
    domain: str
    topic: str
    assimilation: AssimilationReport | None = None
    domain_scores: dict[str, float] = Field(default_factory=dict)
    general_scores: dict[str, float] = Field(default_factory=dict)
    context_scores: dict[str, float] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
