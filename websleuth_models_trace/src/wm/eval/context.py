from __future__ import annotations

import os
import random
import string
import time
from dataclasses import dataclass
from typing import Iterable

from wm.core.schema import BenchmarkExample, BenchmarkResult
from wm.eval.metrics import exact_match, mean
from wm.model.router import HardRouter
from wm.model.wrapper import TraceModel

_FILLER = [
    "The archive contains routine notes about weather, shipping, crops, and local meetings.",
    "Researchers recorded an ordinary observation and moved to the next section.",
    "This paragraph is irrelevant to the requested key and exists only as a distractor.",
    "A committee reviewed the report, filed it, and scheduled a later discussion.",
    "The document continues with background material that does not answer the final question.",
]


def _random_token(rng: random.Random, prefix: str, length: int = 12) -> str:
    return prefix + "_" + "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(length))


def _fill(rng: random.Random, target_words: int) -> list[str]:
    paragraphs: list[str] = []
    count = 0
    while count < target_words:
        sentence = rng.choice(_FILLER)
        paragraphs.append(sentence)
        count += len(sentence.split())
    return paragraphs


def ruler_sniah(length_words: int, rng: random.Random) -> BenchmarkExample:
    key = _random_token(rng, "KEY")
    value = _random_token(rng, "VALUE")
    paragraphs = _fill(rng, length_words)
    index = rng.randrange(len(paragraphs) + 1)
    paragraphs.insert(index, f"The secret pass key for {key} is {value}.")
    context = "\n".join(paragraphs)
    return BenchmarkExample.build(
        prompt=f"{context}\n\nQuestion: What is the secret pass key for {key}?\nAnswer with the value only:",
        answer=value,
        context=context,
        metadata={"task": "ruler_sniah", "length_words": length_words, "needle_position": index / len(paragraphs)},
    )


def ruler_multi_needle(length_words: int, rng: random.Random, count: int = 4) -> BenchmarkExample:
    pairs = [(_random_token(rng, "ITEM", 8), _random_token(rng, "VAL", 8)) for _ in range(count)]
    paragraphs = _fill(rng, length_words)
    for key, value in pairs:
        paragraphs.insert(rng.randrange(len(paragraphs) + 1), f"Registry entry {key} maps exactly to {value}.")
    context = "\n".join(paragraphs)
    keys = ", ".join(key for key, _ in pairs)
    answer = ", ".join(value for _, value in pairs)
    return BenchmarkExample.build(
        prompt=f"{context}\n\nReturn the mapped values for these entries in the same order: {keys}\nAnswer:",
        answer=answer,
        context=context,
        metadata={"task": "ruler_multi_needle", "length_words": length_words, "needles": count},
    )


def ruler_variable_tracking(length_words: int, rng: random.Random) -> BenchmarkExample:
    variables = [f"v{i}" for i in range(8)]
    values = {var: _random_token(rng, "STATE", 6) for var in variables}
    paragraphs = _fill(rng, max(0, length_words - 100))
    assignments = [f"Set {var} = {value}." for var, value in values.items()]
    rng.shuffle(assignments)
    paragraphs.extend(assignments)
    rng.shuffle(paragraphs)
    target = rng.choice(variables)
    context = "\n".join(paragraphs)
    return BenchmarkExample.build(
        prompt=f"{context}\n\nWhat is the final value of {target}? Answer with the value only:",
        answer=values[target],
        context=context,
        metadata={"task": "ruler_variable_tracking", "length_words": length_words},
    )


def ruler_aggregation(length_words: int, rng: random.Random) -> BenchmarkExample:
    marker = _random_token(rng, "MARK", 5)
    occurrences = rng.randint(3, 12)
    paragraphs = _fill(rng, length_words)
    for _ in range(occurrences):
        paragraphs.insert(rng.randrange(len(paragraphs) + 1), f"Aggregation marker: {marker}.")
    context = "\n".join(paragraphs)
    return BenchmarkExample.build(
        prompt=f"{context}\n\nHow many times does the exact marker {marker} appear? Answer with an integer:",
        answer=str(occurrences),
        context=context,
        metadata={"task": "ruler_aggregation", "length_words": length_words},
    )


_BUILDERS = {
    "ruler_sniah": ruler_sniah,
    "ruler_multi_needle": ruler_multi_needle,
    "ruler_variable_tracking": ruler_variable_tracking,
    "ruler_aggregation": ruler_aggregation,
}


def build_ruler_examples(loader: str, max_samples: int, seed: int, length_words: int | None = None) -> list[BenchmarkExample]:
    if loader not in _BUILDERS:
        raise KeyError(loader)
    length = length_words or int(os.environ.get("WM_CONTEXT_LENGTH_WORDS", "4096"))
    return [_BUILDERS[loader](length, random.Random(seed + index * 9973)) for index in range(max(1, max_samples))]


@dataclass
class ContextPoint:
    task: str
    length_words: int
    accuracy: float
    latency_seconds: float
    n: int


class ContextScalingRunner:
    def __init__(self, model: TraceModel, router: HardRouter | None = None):
        self.model = model
        self.router = router

    def run(
        self,
        lengths: Iterable[int] = (2048, 4096, 8192, 16384, 32768),
        tasks: Iterable[str] = tuple(_BUILDERS),
        samples: int = 8,
        seed: int = 42,
    ) -> list[ContextPoint]:
        points: list[ContextPoint] = []
        for task in tasks:
            for length in lengths:
                scores: list[float] = []
                elapsed = 0.0
                for example in build_ruler_examples(task, samples, seed, length_words=length):
                    route = self.router.select(example.prompt).cell_ids if self.router else []
                    start = time.perf_counter()
                    prediction = self.model.generate_text(example.prompt, cell_ids=route, max_new_tokens=64)
                    elapsed += time.perf_counter() - start
                    scores.append(exact_match(prediction, example.answer))
                points.append(ContextPoint(task, length, mean(scores), elapsed / max(samples, 1), samples))
        return points
