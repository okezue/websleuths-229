from __future__ import annotations

import time
from typing import Any, Iterable

from wm.core.schema import BenchmarkExample, BenchmarkResult
from wm.eval.adapters import DatasetUnavailable, load_examples
from wm.eval.code_eval import SandboxedPythonRunner
from wm.eval.ifeval import evaluate_ifeval
from wm.eval.metrics import contains_match, exact_match, mean, numeric_match, token_f1
from wm.eval.registry import BenchmarkCatalog, BenchmarkSpec
from wm.model.router import HardRouter
from wm.model.wrapper import TraceModel


class BenchmarkRunner:
    def __init__(
        self,
        model: TraceModel,
        router: HardRouter,
        catalog: BenchmarkCatalog,
        *,
        max_samples: int = 100,
        seed: int = 42,
        execute_code: bool = False,
    ):
        self.model = model
        self.router = router
        self.catalog = catalog
        self.max_samples = max_samples
        self.seed = seed
        self.execute_code = execute_code
        self.code_runner = SandboxedPythonRunner()
        self._cache: dict[str, list[BenchmarkExample]] = {}

    def examples(self, spec: BenchmarkSpec) -> list[BenchmarkExample]:
        if spec.name not in self._cache:
            self._cache[spec.name] = load_examples(spec, self.max_samples, self.seed)
        return self._cache[spec.name]

    def _multiple_choice(self, example: BenchmarkExample, cell_ids: list[str]) -> tuple[str, float]:
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        scores = [self.model.score_answer(example.prompt, " " + letters[index], cell_ids=cell_ids).nll for index in range(len(example.choices))]
        index = min(range(len(scores)), key=lambda i: scores[i])
        prediction = letters[index]
        return prediction, float(index == example.answer_index)

    def _code(self, example: BenchmarkExample, cell_ids: list[str]) -> tuple[str, float, dict[str, Any]]:
        prediction = self.model.generate_text(example.prompt, cell_ids=cell_ids, max_new_tokens=512)
        if not self.execute_code:
            return prediction, 0.0, {"execution_disabled": True}
        tests = example.metadata.get("test") or example.metadata.get("test_list", [])
        result = self.code_runner.run(prediction, tests)
        return prediction, float(result.passed), result.__dict__

    def evaluate(self, name: str) -> BenchmarkResult:
        spec = self.catalog.get(name)
        start = time.perf_counter()
        try:
            examples = self.examples(spec)
        except DatasetUnavailable as exc:
            return BenchmarkResult(
                name=name,
                group=spec.group,
                metrics={"available": 0.0},
                n=0,
                elapsed_seconds=time.perf_counter() - start,
                metadata={"error": str(exc), "reference": spec.reference},
            )
        rows: list[dict[str, Any]] = []
        primary: list[float] = []
        f1s: list[float] = []
        exacts: list[float] = []
        for example in examples:
            route = self.router.select(example.prompt)
            if example.choices and example.answer_index is not None:
                prediction, score = self._multiple_choice(example, route.cell_ids)
                extra: dict[str, Any] = {}
            elif example.metadata.get("code_task"):
                prediction, score, extra = self._code(example, route.cell_ids)
            else:
                prediction = self.model.generate_text(example.prompt, cell_ids=route.cell_ids, max_new_tokens=256)
                extra = {}
                if spec.metric == "numeric_exact":
                    score = numeric_match(prediction, example.answer)
                elif spec.metric == "instruction_following":
                    score, checks = evaluate_ifeval(
                        prediction,
                        list(example.metadata.get("instruction_id_list", [])),
                        list(example.metadata.get("kwargs", [])),
                    )
                    extra["checks"] = checks
                else:
                    score = max(exact_match(prediction, example.answer), contains_match(prediction, example.answer))
            primary.append(score)
            f1s.append(token_f1(prediction, example.answer) if example.answer else score)
            exacts.append(exact_match(prediction, example.answer) if example.answer else score)
            rows.append(
                {
                    "example_id": example.example_id,
                    "prediction": prediction,
                    "answer": example.answer,
                    "score": score,
                    "route": route.cell_ids,
                    **extra,
                }
            )
        metrics = {
            "primary": mean(primary),
            "accuracy": mean(primary),
            "exact_match": mean(exacts),
            "token_f1": mean(f1s),
            "available": 1.0,
        }
        return BenchmarkResult(
            name=name,
            group=spec.group,
            metrics=metrics,
            n=len(examples),
            elapsed_seconds=time.perf_counter() - start,
            examples=rows,
            metadata={"reference": spec.reference, "metric": spec.metric},
        )

    def evaluate_many(self, names: Iterable[str]) -> dict[str, BenchmarkResult]:
        return {name: self.evaluate(name) for name in names}
