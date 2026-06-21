from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class REPLSession:
    opened_documents: list[str] = field(default_factory=list)
    selected_spans: list[str] = field(default_factory=list)
    extracted_claims: list[str] = field(default_factory=list)
    asserted_claims: list[str] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    request_budget: int = 100
    byte_budget: int = 100_000_000
    requests_used: int = 0
    bytes_used: int = 0

    def consume(self, requests: int = 0, bytes_count: int = 0) -> None:
        if self.requests_used + requests > self.request_budget:
            raise RuntimeError("WebREPL request budget exhausted")
        if self.bytes_used + bytes_count > self.byte_budget:
            raise RuntimeError("WebREPL byte budget exhausted")
        self.requests_used += requests
        self.bytes_used += bytes_count
