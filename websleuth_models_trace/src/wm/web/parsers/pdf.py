from __future__ import annotations

import io

from wm.config import ParseConfig
from wm.core.schema import EvidenceSpan, Locator
from wm.web.parsers.base import ParsedDocument


class PDFParser:
    def __init__(self, cfg: ParseConfig):
        self.cfg = cfg

    def parse(self, *, doc_id: str, url: str, content: bytes, mime_type: str) -> ParsedDocument:
        spans: list[EvidenceSpan] = []
        title = ""
        try:
            import fitz

            pdf = fitz.open(stream=content, filetype="pdf")
            metadata = pdf.metadata or {}
            title = metadata.get("title", "") or ""
            for page_index, page in enumerate(pdf):
                blocks = page.get_text("blocks")
                for block_index, block in enumerate(blocks):
                    x0, y0, x1, y1, text, *_ = block
                    text = " ".join(str(text).split())
                    if not text:
                        continue
                    for start in range(0, len(text), self.cfg.max_text_chars_per_span):
                        part = text[start : start + self.cfg.max_text_chars_per_span]
                        spans.append(
                            EvidenceSpan.build(
                                doc_id=doc_id,
                                modality="pdf",
                                text=part,
                                locator=Locator(page=page_index + 1, bbox=(x0, y0, x1, y1), char_start=start, char_end=start + len(part)),
                                metadata={"block_index": block_index},
                            )
                        )
            pdf.close()
            return ParsedDocument(title=title, spans=spans, metadata={"pages": len({s.locator.page for s in spans})})
        except Exception:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(content))
            title = str((reader.metadata or {}).get("/Title", "") or "")
            for page_index, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                text = " ".join(text.split())
                for start in range(0, len(text), self.cfg.max_text_chars_per_span):
                    part = text[start : start + self.cfg.max_text_chars_per_span]
                    if part:
                        spans.append(
                            EvidenceSpan.build(
                                doc_id=doc_id,
                                modality="pdf",
                                text=part,
                                locator=Locator(page=page_index + 1, char_start=start, char_end=start + len(part)),
                            )
                        )
            return ParsedDocument(title=title, spans=spans, metadata={"pages": len(reader.pages)})
