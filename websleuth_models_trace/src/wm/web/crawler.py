from __future__ import annotations

import logging
from dataclasses import dataclass, field
from urllib.parse import urlparse

from wm.config import FetchConfig, ParseConfig
from wm.core.schema import DocumentRecord
from wm.evidence.store import EvidenceStore
from wm.storage.blob import LocalBlobStore
from wm.web.fetch import Fetcher
from wm.web.frontier import Frontier
from wm.web.parsers.registry import ParserRegistry
from wm.web.policy import canonicalize_url

log = logging.getLogger(__name__)


@dataclass
class CrawlReport:
    fetched: int = 0
    failed: int = 0
    skipped: int = 0
    documents: list[str] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)


class Crawler:
    def __init__(
        self,
        fetch_cfg: FetchConfig,
        parse_cfg: ParseConfig,
        store: EvidenceStore,
        blobs: LocalBlobStore,
    ):
        self.fetch_cfg = fetch_cfg
        self.fetcher = Fetcher(fetch_cfg)
        self.parsers = ParserRegistry(parse_cfg)
        self.store = store
        self.blobs = blobs

    @staticmethod
    def _origin(url: str) -> tuple[str, str]:
        parsed = urlparse(url)
        return parsed.scheme, parsed.netloc

    def _allowed_link(self, seed_origins: set[tuple[str, str]], url: str) -> bool:
        if not self.fetch_cfg.same_origin_only:
            return True
        return self._origin(url) in seed_origins

    def crawl(self, seeds: list[str]) -> CrawlReport:
        report = CrawlReport()
        frontier = Frontier()
        canonical_seeds: list[str] = []
        for seed in seeds:
            try:
                safe = self.fetcher.policy.validate(seed)
            except Exception as exc:
                report.failed += 1
                report.errors.append({"url": seed, "error": str(exc)})
                continue
            canonical_seeds.append(safe)
            frontier.push(safe, priority=1.0, depth=0, reason="seed")
        origins = {self._origin(seed) for seed in canonical_seeds}
        while frontier and report.fetched < self.fetch_cfg.max_pages:
            item = frontier.pop()
            if item.depth > self.fetch_cfg.max_depth:
                report.skipped += 1
                continue
            try:
                fetched = self.fetcher.fetch(item.url)
                blob = self.blobs.put(fetched.content)
                canonical = canonicalize_url(fetched.final_url)
                doc = DocumentRecord.build(
                    url=fetched.url,
                    canonical_url=canonical,
                    title="",
                    mime_type=fetched.mime_type,
                    content_hash=blob.content_hash,
                    blob_path=blob.path,
                    status_code=fetched.status_code,
                    parent_doc_id=item.parent_doc_id,
                    metadata={
                        "headers": fetched.headers,
                        "fetch": fetched.metadata,
                        "depth": item.depth,
                        "reason": item.reason,
                    },
                )
                parsed = self.parsers.parse(
                    doc_id=doc.doc_id,
                    url=canonical,
                    content=fetched.content,
                    mime_type=fetched.mime_type,
                )
                metadata = {**doc.metadata, "parse": {k: v for k, v in parsed.metadata.items() if not isinstance(v, bytes)}}
                if isinstance(parsed.metadata.get("downsampled_bytes"), bytes):
                    down = self.blobs.put(parsed.metadata["downsampled_bytes"])
                    metadata["downsampled_blob_hash"] = down.content_hash
                doc = doc.model_copy(update={"title": parsed.title, "metadata": metadata})
                self.store.put_document(doc)
                self.store.put_spans(parsed.spans)
                report.documents.append(doc.doc_id)
                report.fetched += 1
                for link in parsed.links + parsed.assets:
                    try:
                        safe_link = self.fetcher.policy.validate(link)
                    except Exception:
                        continue
                    if self._allowed_link(origins, safe_link):
                        frontier.push(
                            safe_link,
                            priority=max(0.0, 1.0 - 0.2 * (item.depth + 1)),
                            depth=item.depth + 1,
                            parent_doc_id=doc.doc_id,
                            reason="asset" if link in parsed.assets else "link",
                        )
            except Exception as exc:
                report.failed += 1
                report.errors.append({"url": item.url, "error": f"{type(exc).__name__}: {exc}"})
                log.warning("crawl failed for %s: %s", item.url, exc)
        return report

    def close(self) -> None:
        self.fetcher.close()
