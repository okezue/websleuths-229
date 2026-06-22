from __future__ import annotations

"""Matched continual-learning comparisons across TRACE and mutable baselines."""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from wm.baselines import (
    DPMUBaseline,
    EABSSCBaseline,
    EATRDBaseline,
    ReplaySharedCellBaseline,
    SharedCellBaseline,
)
from wm.config import AppConfig
from wm.core.hashing import stable_hash
from wm.core.io import ensure_dir, write_json
from wm.core.schema import EvidenceEpisode, RouteDecision
from wm.eval.continual import IterativeEvaluator
from wm.eval.metrics import mean
from wm.eval.registry import BenchmarkCatalog
from wm.eval.runner import BenchmarkRunner
from wm.model.wrapper import TraceModel
from wm.pipe.loop import TracePipeline
from wm.reporting.aim_logger import AimLogger
from wm.synthetic.micro_web import load_manifest
from wm.train.residual import exact_match

log = logging.getLogger(__name__)


class RouteProvider:
    """Duck-typed router for evaluating baselines through the native benchmark runner."""

    def __init__(self, provider: Callable[[str], list[str]]):
        self.provider = provider

    def select(self, query: str) -> RouteDecision:
        ids = self.provider(query)
        return RouteDecision(query=query, cell_ids=ids, scores={cell_id: 1.0 for cell_id in ids}, reason="baseline route")


class BaselineMatrixRunner:
    """Run every method from the same initial model and web stream.

    Each method receives a fresh backbone and an isolated evidence/cell store. Domain
    accuracy is measured before and after every episode, and the same fixed general
    benchmark examples are evaluated at configured intervals. The resulting JSON can be
    used directly for paired acquisition/forgetting plots.
    """

    SUPPORTED = ("frozen", "rag", "shared", "replay", "eatrd", "dpmu", "eab_ssc", "trace")

    def __init__(self, cfg: AppConfig, methods: list[str] | None = None):
        self.cfg = cfg
        self.methods = methods or list(self.SUPPORTED)
        unknown = sorted(set(self.methods) - set(self.SUPPORTED))
        if unknown:
            raise ValueError(f"unknown baseline methods: {unknown}")

    def _method_cfg(self, name: str) -> AppConfig:
        root = Path(self.cfg.storage.root) / "baseline_matrix" / name
        storage = self.cfg.storage.model_copy(
            update={
                "root": str(root),
                "database": "evidence.sqlite",
                "blobs": "blobs",
                "cells": "cells",
                "reports": "reports",
            }
        )
        return self.cfg.model_copy(update={"storage": storage}).resolved()

    @staticmethod
    def _prepare(pipeline: TracePipeline, manifest: dict[str, Any], fail_fast: bool) -> list[EvidenceEpisode]:
        episodes: list[EvidenceEpisode] = []
        for item in manifest.get("episodes", []):
            crawl = pipeline.crawl(item["urls"])
            if not crawl.documents:
                message = f"no documents crawled for {item.get('episode_id')}"
                if fail_fast:
                    raise RuntimeError(message)
                log.warning(message)
                continue
            episode = pipeline.compile(
                topic=item.get("topic", item.get("episode_id", "web episode")),
                domain=item.get("domain", "unknown"),
                document_ids=crawl.documents,
            )
            episode = episode.model_copy(update={"metadata": {**episode.metadata, "manifest": item}})
            pipeline.store.put_episode(episode)
            episodes.append(episode)
        return episodes

    @staticmethod
    def _train_qas(episode: EvidenceEpisode):
        return [qa for qa in episode.qa_items if qa.split == "train"]

    def _score_web(
        self,
        model: TraceModel,
        pipeline: TracePipeline,
        episodes: list[EvidenceEpisode],
        route: Callable[[str], list[str]],
        *,
        retrieval: bool = False,
    ) -> float:
        scores: list[float] = []
        for episode in episodes:
            for qa in episode.qa_items:
                if qa.split != "test":
                    continue
                if retrieval:
                    hits = pipeline.index.search(qa.prompt, limit=5)
                    evidence = "\n".join(f"[{hit.span.span_id}] {hit.span.text}" for hit in hits)
                    prompt = f"Use only the verified evidence below.\n{evidence}\n\nQuestion: {qa.prompt}\nAnswer: "
                else:
                    prompt = self.cfg.training.answer_template.format(question=qa.prompt) + " "
                nll = model.score_answer(prompt, qa.answer, cell_ids=route(qa.prompt)).nll
                scores.append(-float(nll))
        return mean(scores)

    def _general_scores(self, model: TraceModel, route: Callable[[str], list[str]]) -> dict[str, float]:
        names = self.cfg.benchmarks.suites.get("general", [])
        if not names:
            return {}
        catalog = BenchmarkCatalog.load(self.cfg.benchmarks.registry)
        runner = BenchmarkRunner(
            model,
            RouteProvider(route),  # type: ignore[arg-type]
            catalog,
            max_samples=self.cfg.benchmarks.max_samples,
            seed=self.cfg.seed,
        )
        results = runner.evaluate_many(names)
        return {name: result.metrics.get("primary", 0.0) for name, result in results.items() if result.n > 0}

    def _run_method(self, name: str, manifest: dict[str, Any]) -> dict[str, Any]:
        cfg = self._method_cfg(name)
        cfg_hash = stable_hash(cfg.model_dump(mode="json"))[:12]
        aim: AimLogger | None = None
        if cfg.aim.enabled:
            aim = AimLogger(
                repo=cfg.aim.repo,
                exp=cfg.aim.experiment or "trace-baseline-matrix",
                name=f"matrix-{name}-{cfg.model.name.replace('/', '_')}-{cfg_hash}",
                hp={**cfg.model_dump(mode="json"), "method": name, "cfg_hash": cfg_hash},
                tags=[
                    f"baseline={name}",
                    f"model={cfg.model.name}",
                    f"seed={cfg.seed}",
                    f"cfg={cfg_hash}",
                    "matrix=baseline_matrix",
                    *cfg.aim.tags,
                ],
            )
        model = TraceModel.from_pretrained(cfg.model)
        pipeline = TracePipeline(cfg, model, aim_logger=aim if name == "trace" else None)
        try:
            episodes = self._prepare(pipeline, manifest, cfg.stream.fail_fast)
            by_domain: dict[str, list[EvidenceEpisode]] = defaultdict(list)
            for episode in episodes:
                by_domain[episode.domain].append(episode)
            learned: set[str] = set()
            continual = IterativeEvaluator()

            learner: Any = None
            if name == "shared":
                learner = SharedCellBaseline(model, rank=cfg.cell.min_rank, lr=cfg.training.learning_rate)
            elif name == "replay":
                learner = ReplaySharedCellBaseline(model, rank=cfg.cell.min_rank, lr=cfg.training.learning_rate, seed=cfg.seed)
            elif name == "eatrd":
                learner = EATRDBaseline(model, rank=cfg.cell.min_rank, lr=cfg.training.learning_rate, seed=cfg.seed)
            elif name == "dpmu":
                learner = DPMUBaseline(model, rank=cfg.cell.min_rank, lr=cfg.training.learning_rate, seed=cfg.seed)
            elif name == "eab_ssc":
                learner = EABSSCBaseline(
                    model,
                    episode_rank=cfg.cell.min_rank,
                    consolidated_rank=cfg.cell.max_rank,
                    lr=cfg.training.learning_rate,
                )

            def route(_query: str) -> list[str]:
                if name == "trace":
                    return pipeline.router.select(_query).cell_ids
                if name in {"shared", "replay"}:
                    return [learner.cell_id]
                if name in {"eatrd", "dpmu"}:
                    return [learner.cell_id]
                if name == "eab_ssc" and learner.consolidated_id in model.cells:
                    return [learner.consolidated_id]
                return []

            retrieval = name == "rag"
            baseline_web = {
                domain: self._score_web(model, pipeline, items, route, retrieval=retrieval)
                for domain, items in by_domain.items()
            }
            continual.set_baseline(baseline_web)
            baseline_general = self._general_scores(model, route)
            if aim and baseline_general:
                aim.set_summary("baseline_general", baseline_general)
            if aim:
                aim.set_summary("baseline_web", baseline_web)
            steps: list[dict[str, Any]] = []
            for index, episode in enumerate(episodes):
                pre = self._score_web(model, pipeline, by_domain[episode.domain], route, retrieval=retrieval)
                continual.record_prelearn(episode.domain, pre)
                train_qas = self._train_qas(episode)
                learn_info: dict[str, Any] = {}
                if name == "trace":
                    learn_info = pipeline.assimilate(episode).model_dump(mode="json")
                elif name == "shared":
                    history = learner.learn(train_qas, steps=cfg.training.max_steps)
                    learn_info = {"steps": len(history), "final_loss": history[-1] if history else 0.0}
                elif name == "replay":
                    history = learner.learn_episode(train_qas, steps=cfg.training.max_steps)
                    learn_info = {"steps": len(history), "final_loss": history[-1] if history else 0.0}
                elif name in {"eatrd", "dpmu"}:
                    learn_info = learner.learn(train_qas, steps=cfg.training.max_steps).__dict__
                elif name == "eab_ssc":
                    episode_cell = learner.learn_episode(train_qas, steps=cfg.training.max_steps)
                    state = learner.consolidate()
                    learn_info = {"episode_cell": episode_cell, **state.__dict__}
                else:
                    learn_info = {"steps": 0}
                learned.add(episode.domain)
                web_scores = {
                    domain: self._score_web(model, pipeline, by_domain[domain], route, retrieval=retrieval)
                    for domain in sorted(learned)
                }
                continual.record_post_step(episode.domain, web_scores)
                if aim and web_scores:
                    aim.track(web_scores, step=index,
                              context={"suite": "web", "episode": episode.episode_id, "domain": episode.domain})
                general_scores: dict[str, float] = {}
                if index % max(cfg.benchmarks.general_every, 1) == 0:
                    general_scores = self._general_scores(model, route)
                    if aim and general_scores:
                        aim.track(general_scores, step=index,
                                  context={"suite": "general", "episode": episode.episode_id})
                if aim:
                    cm = continual.compute().__dict__
                    aim.track({k: float(v) for k, v in cm.items() if isinstance(v, (int, float))},
                              step=index,
                              context={"metric_class": "continual", "episode": episode.episode_id, "domain": episode.domain})
                    aim.track({"n_cell_parameters": float(sum(c.parameter_count for c in model.cells.values())),
                               "n_cells": float(len(model.cells))},
                              step=index,
                              context={"metric_class": "stream_step", "episode": episode.episode_id, "domain": episode.domain})
                steps.append(
                    {
                        "step": index,
                        "episode_id": episode.episode_id,
                        "domain": episode.domain,
                        "pre_score": pre,
                        "web_scores": web_scores,
                        "general_scores": general_scores,
                        "general_deltas": {
                            key: value - baseline_general.get(key, value)
                            for key, value in general_scores.items()
                        },
                        "learn": learn_info,
                        "n_cell_parameters": sum(cell.parameter_count for cell in model.cells.values()),
                        "continual": continual.compute().__dict__,
                    }
                )
            final_continual = continual.compute().__dict__
            if aim:
                aim.set_summary("final_continual", final_continual)
                aim.set_summary("retention_matrix", continual.matrix())
                aim.set_summary("n_episodes", len(steps))
            return {
                "method": name,
                "baseline_web": baseline_web,
                "baseline_general": baseline_general,
                "steps": steps,
                "continual": final_continual,
                "retention_matrix": continual.matrix(),
                "aim_run_hash": aim.hash if aim else None,
            }
        finally:
            try:
                pipeline.close()
            finally:
                if aim:
                    aim.close()

    def run(self, manifest_path: str | Path, output: str | Path | None = None) -> dict[str, Any]:
        manifest = load_manifest(manifest_path)
        results = {name: self._run_method(name, manifest) for name in self.methods}
        summary = {
            name: {
                **result["continual"],
                "final_mean_general_delta": mean(
                    result["steps"][-1].get("general_deltas", {}).values()
                    if result["steps"] else []
                ),
                "final_cell_parameters": (
                    result["steps"][-1].get("n_cell_parameters", 0) if result["steps"] else 0
                ),
            }
            for name, result in results.items()
        }
        report = {
            "manifest": str(Path(manifest_path).resolve()),
            "methods": self.methods,
            "summary": summary,
            "results": results,
        }
        destination = Path(output) if output else Path(self.cfg.storage.reports) / "baseline_matrix.json"
        ensure_dir(destination.parent)
        write_json(destination, report)
        return report
