from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from wm.core.schema import EvidenceSpan


@dataclass
class ParsedDocument:
    title: str = ""
    spans: list[EvidenceSpan] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    assets: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentParser(Protocol):
    def parse(self, *, doc_id: str, url: str, content: bytes, mime_type: str) -> ParsedDocument: ...
