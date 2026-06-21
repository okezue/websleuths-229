from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

from wm.config import ExtractionConfig
from wm.core.schema import ClaimRecord, EvidenceEpisode, EvidenceSpan
from wm.evidence.extraction import RuleClaimExtractor
from wm.evidence.graph import EvidenceGraph
from wm.evidence.questions import QuestionBuilder
from wm.evidence.store import EvidenceStore
from wm.evidence.verifier import EvidenceVerifier
from wm.repl.language import parse_command
from wm.repl.session import REPLSession
from wm.web.index import HybridIndex


class WebREPL:
    """Typed evidence VM. The model can choose operations; only the host executes them."""

    def __init__(
        self,
        store: EvidenceStore,
        index: HybridIndex,
        extraction_cfg: ExtractionConfig,
        *,
        request_budget: int = 100,
        byte_budget: int = 100_000_000,
        seed: int = 42,
    ):
        self.store = store
        self.index = index
        self.extractor = RuleClaimExtractor(extraction_cfg)
        self.verifier = EvidenceVerifier(extraction_cfg)
        self.questions = QuestionBuilder(seed)
        self.session = REPLSession(request_budget=request_budget, byte_budget=byte_budget)

    def seek(self, query: str, n: int = 10, modalities: list[str] | None = None) -> list[dict[str, Any]]:
        self.session.consume(requests=1)
        hits = self.index.search(query, limit=n, modalities=set(modalities) if modalities else None)
        output = []
        for hit in hits:
            doc = self.store.get_document(hit.span.doc_id)
            output.append(
                {
                    "span_id": hit.span.span_id,
                    "doc_id": hit.span.doc_id,
                    "url": doc.canonical_url if doc else "",
                    "title": doc.title if doc else "",
                    "modality": hit.span.modality,
                    "score": hit.score,
                    "preview": hit.span.text[:500],
                }
            )
        self.session.observations.append({"op": "seek", "query": query, "results": output})
        return output

    def open(self, handle: str) -> dict[str, Any]:
        doc = self.store.get_document(handle) or self.store.find_document_by_url(handle)
        if doc is None:
            span = self.store.get_span(handle)
            doc = self.store.get_document(span.doc_id) if span else None
        if doc is None:
            raise KeyError(f"unknown document or span handle: {handle}")
        if doc.doc_id not in self.session.opened_documents:
            self.session.opened_documents.append(doc.doc_id)
        spans = self.store.get_spans(doc.doc_id)
        return {
            "document": doc.model_dump(mode="json"),
            "span_count": len(spans),
            "modalities": sorted({span.modality for span in spans}),
            "preview": "\n".join(span.text for span in spans[:5])[:2000],
        }

    def inspect(
        self,
        handle: str,
        selector: str | None = None,
        channel: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        doc = self.store.get_document(handle)
        if doc:
            spans = self.store.get_spans(doc.doc_id)
        elif span := self.store.get_span(handle):
            spans = [span]
        else:
            raise KeyError(handle)
        if selector:
            spans = [
                span
                for span in spans
                if selector in (span.locator.dom_path or "")
                or selector == span.locator.selector
                or selector in str(span.metadata)
            ]
        if channel:
            spans = [span for span in spans if span.modality == channel]
        spans = spans[:limit]
        for span in spans:
            if span.span_id not in self.session.selected_spans:
                self.session.selected_spans.append(span.span_id)
        return [span.model_dump(mode="json") for span in spans]

    def extract(self, span_ids: Iterable[str] | None = None) -> list[dict[str, Any]]:
        ids = list(span_ids or self.session.selected_spans)
        spans = [span for span_id in ids if (span := self.store.get_span(span_id))]
        doc_ids = {span.doc_id for span in spans}
        documents = {doc_id: doc for doc_id in doc_ids if (doc := self.store.get_document(doc_id))}
        claims = self.extractor.extract(spans, documents)
        verified = self.verifier.verify(claims, {s.span_id: s for s in spans}, documents)
        for claim in verified:
            self.store.put_claim(claim)
            if claim.claim_id not in self.session.extracted_claims:
                self.session.extracted_claims.append(claim.claim_id)
        return [claim.model_dump(mode="json") for claim in verified]

    def assert_claim(
        self,
        span_ids: Iterable[str],
        *,
        subject: str,
        predicate: str,
        object: str,
        text: str | None = None,
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
    ) -> dict[str, Any]:
        claim = ClaimRecord.build(
            subject=subject,
            predicate=predicate,
            object=object,
            text=text,
            support_span_ids=list(span_ids),
            valid_from=valid_from,
            valid_to=valid_to,
        )
        spans = {sid: span for sid in claim.support_span_ids if (span := self.store.get_span(sid))}
        documents = {
            span.doc_id: doc
            for span in spans.values()
            if (doc := self.store.get_document(span.doc_id)) is not None
        }
        verified = self.verifier.verify([claim], spans, documents)[0]
        self.store.put_claim(verified)
        if verified.claim_id not in self.session.asserted_claims:
            self.session.asserted_claims.append(verified.claim_id)
        return verified.model_dump(mode="json")

    def challenge(self, claim_id: str, n: int = 10) -> dict[str, Any]:
        claim = self.store.get_claim(claim_id)
        if not claim:
            raise KeyError(claim_id)
        query = f"{claim.subject} {claim.predicate}"
        hits = self.index.search(query, limit=n)
        counter_spans = [hit.span for hit in hits if claim.object.lower() not in hit.span.text.lower()]
        all_claims = self.extract([span.span_id for span in counter_spans]) if counter_spans else []
        contradictions = [
            item
            for item in all_claims
            if item["subject"].lower() == claim.subject.lower()
            and item["predicate"].lower() == claim.predicate.lower()
            and item["object"].lower() != claim.object.lower()
        ]
        return {
            "claim": claim.model_dump(mode="json"),
            "known_contradictions": [self.store.get_claim(cid).model_dump(mode="json") for cid in claim.contradicts if self.store.get_claim(cid)],
            "candidate_counterevidence": contradictions,
        }

    def link(self, src: str, dst: str, relation: str, weight: float = 1.0) -> dict[str, Any]:
        self.store.put_edge(src, dst, relation, weight)
        return {"src": src, "dst": dst, "relation": relation, "weight": weight}

    def ask(self, question: str, at_time: datetime | None = None) -> dict[str, Any]:
        graph = EvidenceGraph(self.store.get_claims())
        answer = graph.answer(question, when=at_time)
        return answer.__dict__ if answer else {"answer": None, "claim_ids": [], "proof_span_ids": [], "confidence": 0.0}

    def commit(
        self,
        *,
        topic: str,
        domain: str,
        claim_ids: Iterable[str] | None = None,
    ) -> EvidenceEpisode:
        ids = list(claim_ids or self.session.asserted_claims or self.session.extracted_claims)
        claims = [claim for cid in ids if (claim := self.store.get_claim(cid))]
        span_ids = sorted({sid for claim in claims for sid in claim.support_span_ids})
        document_ids = sorted({self.store.get_span(sid).doc_id for sid in span_ids if self.store.get_span(sid)})
        episode = EvidenceEpisode.build(
            topic=topic,
            domain=domain,
            claim_ids=[claim.claim_id for claim in claims],
            span_ids=span_ids,
            document_ids=document_ids,
            qa_items=self.questions.build(claims),
            metadata={
                "repl": {
                    "requests_used": self.session.requests_used,
                    "bytes_used": self.session.bytes_used,
                    "observations": self.session.observations,
                }
            },
        )
        self.store.put_episode(episode)
        return episode

    def execute(self, line: str) -> Any:
        command = parse_command(line)
        if command.op == "noop":
            return None
        aliases = {"assert": "assert_claim"}
        method = getattr(self, aliases.get(command.op, command.op), None)
        if not callable(method):
            raise ValueError(f"unknown WebREPL operation: {command.op}")
        return method(*command.args, **command.kwargs)

    def execute_script(self, script: str) -> list[Any]:
        output: list[Any] = []
        for line in script.splitlines():
            result = self.execute(line)
            if result is not None:
                output.append(result)
        return output
