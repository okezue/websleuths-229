from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from wm.config import AppConfig
from wm.core.io import ensure_dir, write_json
from wm.core.schema import BenchmarkResult, EvidenceEpisode, StreamStepReport
from wm.eval.continual import IterativeEvaluator
from wm.eval.metrics import mean
from wm.eval.registry import BenchmarkCatalog
from wm.eval.runner import BenchmarkRunner
from wm.model.wrapper import TraceModel
from wm.pipe.loop import TracePipeline
from wm.synthetic.micro_web import load_manifest
from wm.train.residual import exact_match

log = logging.getLogger(__name__)


class StreamRunner:
    def __init__(self, cfg: AppConfig, model: TraceModel | None = None):
        self.cfg = cfg
        self.model = model or TraceModel.from_pretrained(cfg.model)
        self.pipeline = TracePipeline(cfg, self.model)
        self.catalog = BenchmarkCatalog.load(cfg.benchmarks.registry)
        self.bench = BenchmarkRunner(
            self.model,
            self.pipeline.router,
            self.catalog,
            max_samples=cfg.benchmarks.max_samples,
            seed=cfg.seed,
        )
        self.continual = IterativeEvaluator()
        self.reports_dir = ensure_dir(cfg.storage.reports)

    def _episode_score(self, episodes: list[EvidenceEpisode]) -> float:
        scores: list[float] = []
        for episode in episodes:
            for qa in episode.qa_items:
                if qa.split != "test":
                    continue
                route = self.pipeline.router.select(qa.prompt).cell_ids
                prompt = self.cfg.training.answer_template.format(question=qa.prompt) + " "
                prediction = self.model.generate_text(prompt, cell_ids=route, max_new_tokens=48)
                scores.append(exact_match(prediction, qa.answer))
        return mean(scores)

    def _benchmark_scores(self, names: list[str]) -> tuple[dict[str, float], dict[str, BenchmarkResult]]:
        results = self.bench.evaluate_many(names)
        scores = {name: result.metrics.get("primary", 0.0) for name, result in results.items() if result.n > 0}
        return scores, results

    def _prepare_episodes(self, manifest: dict[str, Any]) -> list[EvidenceEpisode]:
        prepared: list[EvidenceEpisode] = []
        for item in manifest.get("episodes", []):
            crawl = self.pipeline.crawl(item["urls"])
            if not crawl.documents:
                message = f"no documents crawled for {item.get('episode_id')}"
                if self.cfg.stream.fail_fast:
                    raise RuntimeError(message)
                log.warning(message)
                continue
            episode = self.pipeline.compile(
                topic=item.get("topic", item.get("episode_id", "web episode")),
                domain=item.get("domain", "unknown"),
                document_ids=crawl.documents,
            )
            episode = episode.model_copy(update={"metadata": {**episode.metadata, "manifest": item}})
            self.pipeline.store.put_episode(episode)
            prepared.append(episode)
        return prepared

    def run(self, manifest_path: str | None = None) -> dict[str, Any]:
        path = manifest_path or self.cfg.stream.manifest
        if not path:
            raise ValueError("stream.manifest is required")
        manifest = load_manifest(path)
        episodes = self._prepare_episodes(manifest)
        by_domain: dict[str, list[EvidenceEpisode]] = defaultdict(list)
        for episode in episodes:
            by_domain[episode.domain].append(episode)
        baseline_web = {domain: self._episode_score(items) for domain, items in by_domain.items()}
        self.continual.set_baseline(baseline_web)
        domain_names = self.cfg.benchmarks.suites.get("domain", [])
        general_names = self.cfg.benchmarks.suites.get("general", [])
        context_names = self.cfg.benchmarks.suites.get("long_context", [])
        baseline_external, baseline_external_results = self._benchmark_scores(domain_names + general_names)
        baseline_context, baseline_context_results = self._benchmark_scores(context_names) if context_names else ({}, {})
        steps: list[StreamStepReport] = []
        learned_domains: set[str] = set()
        for index, episode in enumerate(episodes):
            pre_score = self._episode_score(by_domain[episode.domain])
            self.continual.record_prelearn(episode.domain, pre_score)
            assimilation = self.pipeline.assimilate(episode)
            learned_domains.add(episode.domain)
            web_scores = {
                domain: self._episode_score(by_domain[domain])
                for domain in sorted(learned_domains if self.cfg.stream.evaluate_all_prior_domains else {episode.domain})
            }
            self.continual.record_post_step(episode.domain, web_scores)
            external_domain_scores: dict[str, float] = {}
            domain_results: dict[str, BenchmarkResult] = {}
            if domain_names:
                relevant = [
                    name
                    for name in domain_names
                    if self.catalog.get(name).group in learned_domains or self.catalog.get(name).group == episode.domain
                ]
                external_domain_scores, domain_results = self._benchmark_scores(relevant)
            general_scores: dict[str, float] = {}
            general_results: dict[str, BenchmarkResult] = {}
            if general_names and index % max(self.cfg.benchmarks.general_every, 1) == 0:
                general_scores, general_results = self._benchmark_scores(general_names)
            context_scores: dict[str, float] = {}
            context_results: dict[str, BenchmarkResult] = {}
            if context_names and index % max(self.cfg.benchmarks.context_every, 1) == 0:
                context_scores, context_results = self._benchmark_scores(context_names)
            continual_metrics = self.continual.compute().__dict__
            general_deltas = {
                name: score - baseline_external.get(name, score)
                for name, score in general_scores.items()
            }
            domain_deltas = {
                name: score - baseline_external.get(name, score)
                for name, score in external_domain_scores.items()
            }
            context_deltas = {
                name: score - baseline_context.get(name, score)
                for name, score in context_scores.items()
            }
            report = StreamStepReport(
                step=index,
                episode_id=episode.episode_id,
                domain=episode.domain,
                topic=episode.topic,
                assimilation=assimilation,
                domain_scores={**{f"web:{domain}": score for domain, score in web_scores.items()}, **external_domain_scores},
                general_scores=general_scores,
                context_scores=context_scores,
                metrics={
                    **continual_metrics,
                    "n_cells": float(len(self.model.cells)),
                    "total_cell_parameters": float(sum(cell.parameter_count for cell in self.model.cells.values())),
                    "current_pre_score": pre_score,
                    "current_post_score": web_scores.get(episode.domain, 0.0),
                    "mean_general_delta": mean(general_deltas.values()),
                    "worst_general_delta": min(general_deltas.values(), default=0.0),
                    "general_improvement_count": float(sum(delta > 0 for delta in general_deltas.values())),
                    "mean_domain_benchmark_delta": mean(domain_deltas.values()),
                    "mean_context_delta": mean(context_deltas.values()),
                },
            )
            steps.append(report)
            if self.cfg.stream.save_every_step:
                write_json(
                    self.reports_dir / f"step_{index:04d}.json",
                    {
                        "step": report.model_dump(mode="json"),
                        "benchmark_details": {
                            **{name: result.model_dump(mode="json") for name, result in domain_results.items()},
                            **{name: result.model_dump(mode="json") for name, result in general_results.items()},
                            **{name: result.model_dump(mode="json") for name, result in context_results.items()},
                        },
                    },
                )
        final = {
            "config": self.cfg.model_dump(mode="json"),
            "manifest": str(Path(path).resolve()),
            "baseline_web": baseline_web,
            "baseline_external": baseline_external,
            "baseline_external_details": {name: result.model_dump(mode="json") for name, result in baseline_external_results.items()},
            "baseline_context": baseline_context,
            "baseline_context_details": {name: result.model_dump(mode="json") for name, result in baseline_context_results.items()},
            "steps": [step.model_dump(mode="json") for step in steps],
            "continual": self.continual.compute().__dict__,
            "retention_matrix": self.continual.matrix(),
        }
        write_json(self.reports_dir / "stream_report.json", final)
        return final

    def close(self) -> None:
        self.pipeline.close()
