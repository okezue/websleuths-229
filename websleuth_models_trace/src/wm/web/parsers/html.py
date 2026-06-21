from __future__ import annotations

import json
import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag

from wm.config import ParseConfig
from wm.core.schema import EvidenceSpan, Locator
from wm.web.parsers.base import ParsedDocument


_DROP_TAGS = {
    "script",
    "style",
    "noscript",
    "template",
    "svg",
    "canvas",
    "nav",
    "footer",
    "header",
    "aside",
    "form",
}
_TEXT_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote", "pre", "code", "figcaption", "dt", "dd"}


def _dom_path(tag: Tag) -> str:
    parts: list[str] = []
    current: Tag | None = tag
    while current and current.name not in {"[document]", "html"}:
        if not current.name:
            break
        index = 1
        if current.parent:
            siblings = [s for s in current.parent.find_all(current.name, recursive=False)]
            if current in siblings:
                index = siblings.index(current) + 1
        parts.append(f"{current.name}[{index}]")
        current = current.parent if isinstance(current.parent, Tag) else None
    return "/" + "/".join(reversed(parts))


def _clean(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class HTMLParser:
    def __init__(self, cfg: ParseConfig):
        self.cfg = cfg

    def parse(self, *, doc_id: str, url: str, content: bytes, mime_type: str) -> ParsedDocument:
        soup = BeautifulSoup(content, "lxml")
        title = _clean(soup.title.get_text(" ", strip=True)) if soup.title else ""
        source_meta = soup.find("meta", attrs={"name": "trace-source-domain"})
        source_domain = str(source_meta.get("content", "")) if source_meta else ""
        json_ld: list[object] = []
        for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
            try:
                json_ld.append(json.loads(script.string or script.get_text()))
            except (json.JSONDecodeError, TypeError):
                pass
        trace_claims: list[dict] = []
        for tag in soup.select("[data-trace-claim]"):
            try:
                trace_claims.append(json.loads(tag.get("data-trace-claim", "{}")))
            except json.JSONDecodeError:
                pass
        for name in _DROP_TAGS:
            for tag in soup.find_all(name):
                tag.decompose()
        links: list[str] = []
        assets: list[str] = []
        for a in soup.find_all("a", href=True):
            target = urljoin(url, str(a["href"]))
            if target.startswith(("http://", "https://", "file://")):
                links.append(target)
        for tag, attr in [("img", "src"), ("video", "src"), ("audio", "src"), ("source", "src")]:
            for node in soup.find_all(tag):
                if node.get(attr):
                    assets.append(urljoin(url, str(node[attr])))
        spans: list[EvidenceSpan] = []
        char_offset = 0
        for tag in soup.find_all(list(_TEXT_TAGS)):
            text = _clean(tag.get_text(" ", strip=True))
            if len(text) < 2:
                continue
            modality = "code" if tag.name in {"pre", "code"} else "html"
            for start in range(0, len(text), self.cfg.max_text_chars_per_span):
                part = text[start : start + self.cfg.max_text_chars_per_span]
                spans.append(
                    EvidenceSpan.build(
                        doc_id=doc_id,
                        modality=modality,
                        text=part,
                        locator=Locator(
                            dom_path=_dom_path(tag),
                            selector=tag.name,
                            char_start=char_offset + start,
                            char_end=char_offset + start + len(part),
                        ),
                        metadata={
                            "tag": tag.name,
                            "id": tag.get("id"),
                            "classes": tag.get("class", []),
                            "heading": tag.name.startswith("h"),
                        },
                    )
                )
            char_offset += len(text) + 1
        for table_index, table in enumerate(soup.find_all("table")):
            rows = table.find_all("tr")
            for row_index, row in enumerate(rows):
                cells = [_clean(cell.get_text(" ", strip=True)) for cell in row.find_all(["th", "td"])]
                if not any(cells):
                    continue
                spans.append(
                    EvidenceSpan.build(
                        doc_id=doc_id,
                        modality="table",
                        text=" | ".join(cells),
                        locator=Locator(dom_path=_dom_path(table), table_row=row_index),
                        metadata={"table_index": table_index, "cells": cells},
                    )
                )
        return ParsedDocument(
            title=title,
            spans=spans,
            links=list(dict.fromkeys(links)),
            assets=list(dict.fromkeys(assets)),
            metadata={"json_ld": json_ld, "trace_claims": trace_claims, "source_domain": source_domain},
        )
