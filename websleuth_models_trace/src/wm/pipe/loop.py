from __future__ import annotations

from typing import Iterable

from wm.config import AppConfig
from wm.core.schema import AssimilationReport, EvidenceEpisode
from wm.evidence.compiler import EvidenceCompiler
from wm.evidence.store import EvidenceStore
from wm.model.bank import CellBank
from wm.model.router import HardRouter
from wm.model.wrapper import TraceModel
from wm.reporting.aim_logger import AimLogger
from wm.storage.blob import LocalBlobStore
from wm.train.assimilator import Assimilator
from wm.web.crawler import Crawler, CrawlReport
from wm.web.index import HybridIndex


class TracePipeline:
    def __init__(self, cfg: AppConfig, model: TraceModel | None = None, aim_logger: AimLogger | None = None):
        self.cfg = cfg
        self.aim = aim_logger
        self.store = EvidenceStore(cfg.storage.database)
        self.blobs = LocalBlobStore(cfg.storage.blobs)
        self.index = HybridIndex(self.store)
        self.compiler = EvidenceCompiler(self.store, cfg.extraction, cfg.seed)
        self.model = model
        self.router = HardRouter(
            threshold=cfg.cell.route_threshold,
            max_active=cfg.cell.max_active_cells,
            entity_bonus=cfg.cell.route_entity_bonus,
        )
        self.bank = CellBank(cfg.storage.cells)
        if model:
            for cell, metadata in self.bank.load_all(
                model.hidden_size,
                dropout=cfg.cell.dropout,
                scale=cfg.cell.scale,
            ):
                model.add_cell(cell, trainable=False)
                self.router.add(metadata)
            self.assimilator = Assimilator(cfg, model, self.router, self.bank, self.store, aim_logger=self.aim)
        else:
            self.assimilator = None

    def crawl(self, urls: Iterable[str]) -> CrawlReport:
        crawler = Crawler(self.cfg.fetch, self.cfg.parse, self.store, self.blobs)
        try:
            report = crawler.crawl(list(urls))
        finally:
            crawler.close()
        self.index.rebuild()
        return report

    def compile(self, *, topic: str, domain: str, document_ids: Iterable[str] | None = None) -> EvidenceEpisode:
        return self.compiler.compile(topic=topic, domain=domain, document_ids=document_ids)

    def assimilate(self, episode: EvidenceEpisode) -> AssimilationReport:
        if self.assimilator is None:
            raise RuntimeError("TracePipeline was created without a model")
        return self.assimilator.assimilate(episode)

    def answer(self, prompt: str, use_retrieval: bool = False, top_k: int = 5) -> str:
        if self.model is None:
            raise RuntimeError("TracePipeline was created without a model")
        route = self.router.select(prompt)
        if use_retrieval:
            evidence = "\n".join(hit.span.text for hit in self.index.search(prompt, limit=top_k))
            prompt = f"Verified evidence:\n{evidence}\n\nQuestion: {prompt}\nAnswer:"
        return self.model.generate_text(prompt, cell_ids=route.cell_ids)

    def close(self) -> None:
        if self.model:
            self.model.close()
        self.store.close()


# Compatibility name for the original repository's central pipeline.
AgenticPipeline = TracePipeline
