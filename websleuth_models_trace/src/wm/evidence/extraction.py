from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Iterable

from wm.config import ExtractionConfig
from wm.core.schema import ClaimRecord, DocumentRecord, EvidenceSpan

log = logging.getLogger(__name__)

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_COPULA_RE = re.compile(
    r"^(?P<subject>[A-Z0-9][^.!?]{1,100}?)\s+(?P<predicate>is|are|was|were|equals|equaled|became|remains|contains|includes|has|had)\s+(?P<object>[^.!?]{2,220})[.!?]?$",
    re.IGNORECASE,
)
_TRANSITIVE_RE = re.compile(
    r"^(?P<subject>[A-Z0-9][^.!?]{1,100}?)\s+(?P<predicate>acquired|approved|announced|reported|measured|observed|discovered|produced|reduced|increased|decreased|caused|requires|supports|prohibits|allows|treats|targets|released|won|founded|launched)\s+(?P<object>[^.!?]{2,220})[.!?]?$",
    re.IGNORECASE,
)
_KEY_VALUE_RE = re.compile(r"^(?P<subject>[^:|]{2,100})\s*[:|]\s*(?P<predicate>[^:|]{2,80})\s*[:|]\s*(?P<object>[^:|]{1,220})$")
_TRACE_LINE_RE = re.compile(r"TRACE_CLAIM\s*[:=]\s*(\{.*\})", re.DOTALL)


class ClaimExtractor(ABC):
    @abstractmethod
    def extract(self, spans: Iterable[EvidenceSpan], documents: dict[str, DocumentRecord]) -> list[ClaimRecord]: ...


class RuleClaimExtractor(ClaimExtractor):
    def __init__(self, cfg: ExtractionConfig):
        self.cfg = cfg

    def _from_mapping(self, mapping: dict[str, Any], support: list[str]) -> ClaimRecord | None:
        subject = str(mapping.get("subject", "")).strip()
        predicate = str(mapping.get("predicate", "")).strip()
        obj = str(mapping.get("object", mapping.get("value", ""))).strip()
        if not subject or not predicate or not obj:
            return None
        qualifiers = dict(mapping.get("qualifiers", {}))
        return ClaimRecord.build(
            subject=subject,
            predicate=predicate,
            object=obj,
            text=str(mapping.get("text") or f"{subject} {predicate} {obj}."),
            support_span_ids=support,
            valid_from=mapping.get("valid_from"),
            valid_to=mapping.get("valid_to"),
            confidence=float(mapping.get("confidence", 0.9)),
            qualifiers=qualifiers,
        )

    def _metadata_claims(self, documents: dict[str, DocumentRecord], spans: list[EvidenceSpan]) -> list[ClaimRecord]:
        by_doc: dict[str, list[EvidenceSpan]] = {}
        for span in spans:
            by_doc.setdefault(span.doc_id, []).append(span)
        claims: list[ClaimRecord] = []
        for doc_id, doc in documents.items():
            parsed = doc.metadata.get("parse", {}) if isinstance(doc.metadata, dict) else {}
            trace_claims = parsed.get("trace_claims", []) if isinstance(parsed, dict) else []
            for item in trace_claims:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text") or "")
                candidates = by_doc.get(doc_id, [])
                support = [s.span_id for s in candidates if text and text.lower() in s.text.lower()]
                if not support and candidates:
                    support = [candidates[0].span_id]
                claim = self._from_mapping(item, support)
                if claim:
                    claims.append(claim)
        return claims

    def _sentence_claim(self, sentence: str, span_id: str) -> ClaimRecord | None:
        sentence = " ".join(sentence.split()).strip()
        if not (self.cfg.min_claim_chars <= len(sentence) <= self.cfg.max_claim_chars):
            return None
        trace = _TRACE_LINE_RE.search(sentence)
        if trace:
            try:
                return self._from_mapping(json.loads(trace.group(1)), [span_id])
            except json.JSONDecodeError:
                return None
        for pattern in (_KEY_VALUE_RE, _COPULA_RE, _TRANSITIVE_RE):
            match = pattern.match(sentence)
            if match:
                groups = match.groupdict()
                return ClaimRecord.build(
                    subject=groups["subject"].strip(" -"),
                    predicate=groups["predicate"].strip(" -"),
                    object=groups["object"].strip(" -"),
                    text=sentence,
                    support_span_ids=[span_id],
                    confidence=0.55,
                )
        return None

    def extract(self, spans: Iterable[EvidenceSpan], documents: dict[str, DocumentRecord]) -> list[ClaimRecord]:
        span_list = list(spans)
        claims = self._metadata_claims(documents, span_list)
        for span in span_list:
            for sentence in _SENTENCE_RE.split(span.text):
                if claim := self._sentence_claim(sentence, span.span_id):
                    claims.append(claim)
        merged: dict[str, ClaimRecord] = {}
        for claim in claims:
            if claim.claim_id in merged:
                old = merged[claim.claim_id]
                merged[claim.claim_id] = old.model_copy(
                    update={
                        "support_span_ids": list(dict.fromkeys(old.support_span_ids + claim.support_span_ids)),
                        "confidence": max(old.confidence, claim.confidence),
                    }
                )
            else:
                merged[claim.claim_id] = claim
        return list(merged.values())


class LocalLLMClaimExtractor(ClaimExtractor):
    """Optional local-only structured extractor. It never has network or tool access."""

    def __init__(self, cfg: ExtractionConfig):
        if not cfg.local_model:
            raise ValueError("extraction.local_model is required for local_llm backend")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install websleuth-models[models]") from exc
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.local_model)
        self.model = AutoModelForCausalLM.from_pretrained(cfg.local_model)
        self.model.eval()

    def _parse_json(self, text: str) -> list[dict[str, Any]]:
        start = text.find("[")
        end = text.rfind("]") + 1
        if start < 0 or end <= start:
            return []
        try:
            value = json.loads(text[start:end])
        except json.JSONDecodeError:
            return []
        return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []

    def extract(self, spans: Iterable[EvidenceSpan], documents: dict[str, DocumentRecord]) -> list[ClaimRecord]:
        import torch

        claims: list[ClaimRecord] = []
        for span in spans:
            if len(span.text) < self.cfg.min_claim_chars:
                continue
            prompt = (
                "Extract only claims directly stated in the SOURCE. Return a JSON array of objects with "
                "subject, predicate, object, text. Do not use outside knowledge.\nSOURCE:\n" + span.text[:6000] + "\nJSON:"
            )
            tokens = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            with torch.no_grad():
                output = self.model.generate(**tokens, max_new_tokens=512, do_sample=False)
            generated = self.tokenizer.decode(output[0][tokens["input_ids"].shape[1] :], skip_special_tokens=True)
            for item in self._parse_json(generated):
                subject = str(item.get("subject", "")).strip()
                predicate = str(item.get("predicate", "")).strip()
                obj = str(item.get("object", "")).strip()
                text = str(item.get("text", "")).strip()
                if subject and predicate and obj and text and text.lower() in span.text.lower():
                    claims.append(
                        ClaimRecord.build(
                            subject=subject,
                            predicate=predicate,
                            object=obj,
                            text=text,
                            support_span_ids=[span.span_id],
                            confidence=0.6,
                        )
                    )
        return claims


class HybridClaimExtractor(ClaimExtractor):
    def __init__(self, cfg: ExtractionConfig):
        self.rules = RuleClaimExtractor(cfg)
        self.local = LocalLLMClaimExtractor(cfg) if cfg.local_model else None

    def extract(self, spans: Iterable[EvidenceSpan], documents: dict[str, DocumentRecord]) -> list[ClaimRecord]:
        span_list = list(spans)
        claims = self.rules.extract(span_list, documents)
        if self.local:
            claims.extend(self.local.extract(span_list, documents))
        merged: dict[str, ClaimRecord] = {}
        for claim in claims:
            if claim.claim_id not in merged:
                merged[claim.claim_id] = claim
            else:
                old = merged[claim.claim_id]
                merged[claim.claim_id] = old.model_copy(
                    update={"support_span_ids": list(dict.fromkeys(old.support_span_ids + claim.support_span_ids))}
                )
        return list(merged.values())


def build_extractor(cfg: ExtractionConfig) -> ClaimExtractor:
    if cfg.backend == "rules":
        return RuleClaimExtractor(cfg)
    if cfg.backend == "local_llm":
        return LocalLLMClaimExtractor(cfg)
    return HybridClaimExtractor(cfg)
