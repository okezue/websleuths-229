from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from wm.config import ExtractionConfig
from wm.core.schema import ClaimRecord, DocumentRecord, EvidenceEpisode, EvidenceSpan
from wm.evidence.extraction import build_extractor
from wm.evidence.questions import QuestionBuilder
from wm.evidence.store import EvidenceStore
from wm.evidence.verifier import EvidenceVerifier


class EvidenceCompiler:
    def __init__(self, store: EvidenceStore, cfg: ExtractionConfig, seed: int = 42):
        self.store = store
        self.extractor = build_extractor(cfg)
        self.verifier = EvidenceVerifier(cfg)
        self.questions = QuestionBuilder(seed)

    def compile(
        self,
        *,
        topic: str,
        domain: str,
        document_ids: Iterable[str] | None = None,
        include_candidates: bool = False,
    ) -> EvidenceEpisode:
        doc_ids = list(document_ids) if document_ids is not None else [doc.doc_id for doc in self.store.list_documents()]
        documents: dict[str, DocumentRecord] = {
            doc_id: doc for doc_id in doc_ids if (doc := self.store.get_document(doc_id)) is not None
        }
        spans_list = [span for doc_id in documents for span in self.store.get_spans(doc_id)]
        spans: dict[str, EvidenceSpan] = {span.span_id: span for span in spans_list}
        extracted = self.extractor.extract(spans_list, documents)
        verified = self.verifier.verify(extracted, spans, documents)
        learnable_statuses = {"supported"}
        if include_candidates:
            learnable_statuses.update({"candidate", "contested"})
        selected = [claim for claim in verified if claim.status in learnable_statuses]
        self.store.put_claims(verified)
        for claim in verified:
            for span_id in claim.support_span_ids:
                self.store.put_edge(claim.claim_id, span_id, "supported_by", claim.confidence)
            for other in claim.contradicts:
                self.store.put_edge(claim.claim_id, other, "contradicts", 1.0)
        qa_items = self.questions.build(selected)
        episode = EvidenceEpisode.build(
            topic=topic,
            domain=domain,
            claim_ids=[claim.claim_id for claim in selected],
            document_ids=sorted(documents),
            span_ids=sorted(spans),
            qa_items=qa_items,
            metadata={
                "n_extracted": len(extracted),
                "n_verified": len(selected),
                "status_counts": dict(
                    (lambda counts: counts)(
                        defaultdict(int, {status: sum(1 for c in verified if c.status == status) for status in {c.status for c in verified}})
                    )
                ),
            },
        )
        self.store.put_episode(episode)
        return episode
