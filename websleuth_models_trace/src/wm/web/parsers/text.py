from __future__ import annotations

import re

from wm.config import ParseConfig
from wm.core.schema import EvidenceSpan, Locator
from wm.web.parsers.base import ParsedDocument


class TextParser:
    def __init__(self, cfg: ParseConfig):
        self.cfg = cfg

    def parse(self, *, doc_id: str, url: str, content: bytes, mime_type: str) -> ParsedDocument:
        text = content.decode("utf-8", errors="replace").replace("\x00", "")
        chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n", text) if chunk.strip()]
        spans: list[EvidenceSpan] = []
        offset = 0
        for chunk in chunks:
            for start in range(0, len(chunk), self.cfg.max_text_chars_per_span):
                part = chunk[start : start + self.cfg.max_text_chars_per_span]
                spans.append(
                    EvidenceSpan.build(
                        doc_id=doc_id,
                        modality="text",
                        text=part,
                        locator=Locator(char_start=offset + start, char_end=offset + start + len(part)),
                    )
                )
            offset += len(chunk) + 2
        return ParsedDocument(spans=spans)
