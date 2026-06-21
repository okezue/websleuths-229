from __future__ import annotations

"""Optional adapter for EleutherAI's lm-evaluation-harness.

The harness is deliberately imported lazily so the core repository remains usable without
its heavier optional dependency. The adapter routes each request through TRACE's hard
router and therefore measures the deployed backbone-plus-cell system rather than the naked
backbone.
"""

from collections.abc import Iterable
from typing import Any

import torch

from wm.model.router import HardRouter
from wm.model.wrapper import TraceModel


class LMEvalUnavailable(RuntimeError):
    """Raised when the optional ``lm-eval`` package is not installed."""


def _request_args(request: Any) -> tuple[Any, ...]:
    args = getattr(request, "args", request)
    if isinstance(args, tuple):
        return args
    if isinstance(args, list):
        return tuple(args)
    return (args,)


def build_lm_eval_adapter(model: TraceModel, router: HardRouter):
    """Return an ``lm_eval.api.model.LM`` instance backed by a :class:`TraceModel`.

    The function avoids binding to lm-eval at package import time. It supports the request
    shapes used by lm-eval 0.4.x and keeps the implementation intentionally serial; the
    repository's native benchmark runner should be used when detailed per-example routes
    and provenance are required.
    """

    try:
        from lm_eval.api.model import LM
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise LMEvalUnavailable("Install websleuth-models[eval]") from exc

    class TraceHarnessLM(LM):
        def __init__(self) -> None:
            super().__init__()
            self.rank = 0
            self.world_size = 1

        @property
        def eot_token_id(self) -> int:
            token_id = getattr(model.tokenizer, "eos_token_id", None)
            return int(token_id if token_id is not None else 0)

        @property
        def max_length(self) -> int:
            return int(model.max_input_tokens)

        @property
        def max_gen_toks(self) -> int:
            return int(model.max_new_tokens)

        @property
        def batch_size(self) -> int:
            return 1

        @property
        def device(self) -> torch.device:
            return model.device

        @property
        def tokenizer_name(self) -> str:
            return type(model.tokenizer).__name__

        def tok_encode(self, string: str, **_: Any) -> list[int]:
            return list(model.tokenizer(string, add_special_tokens=False)["input_ids"])

        def tok_decode(self, tokens: Iterable[int], **_: Any) -> str:
            return str(model.tokenizer.decode(list(tokens), skip_special_tokens=True))

        def loglikelihood(self, requests: list[Any]) -> list[tuple[float, bool]]:
            results: list[tuple[float, bool]] = []
            for request in requests:
                args = _request_args(request)
                context = str(args[0])
                continuation = str(args[1])
                cell_ids = router.select(context + continuation).cell_ids
                score = model.score_answer(context, continuation, cell_ids=cell_ids)
                logits = model.answer_logits(context, continuation, cell_ids=cell_ids)
                target_ids = model.tokenizer(continuation, add_special_tokens=False)["input_ids"]
                length = min(logits.shape[1], len(target_ids))
                greedy = True
                if length:
                    predictions = logits[0, :length].argmax(dim=-1).detach().cpu().tolist()
                    greedy = predictions == list(target_ids)[:length]
                results.append((-score.nll * score.token_count, greedy))
            return results

        def loglikelihood_rolling(self, requests: list[Any]) -> list[float]:
            values: list[float] = []
            for request in requests:
                text = str(_request_args(request)[0])
                # Use a one-byte prefix so answer scoring has a well-defined previous token.
                cell_ids = router.select(text).cell_ids
                score = model.score_answer(" ", text, cell_ids=cell_ids)
                values.append(-score.nll * score.token_count)
            return values

        def generate_until(self, requests: list[Any]) -> list[str]:
            outputs: list[str] = []
            for request in requests:
                args = _request_args(request)
                context = str(args[0])
                generation = args[1] if len(args) > 1 and isinstance(args[1], dict) else {}
                until = generation.get("until", [])
                if isinstance(until, str):
                    until = [until]
                max_tokens = int(generation.get("max_gen_toks", self.max_gen_toks))
                temperature = float(generation.get("temperature", 0.0) or 0.0)
                cell_ids = router.select(context).cell_ids
                text = model.generate_text(
                    context,
                    cell_ids=cell_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                )
                for stop in until:
                    if stop and stop in text:
                        text = text.split(stop, 1)[0]
                outputs.append(text)
            return outputs

    return TraceHarnessLM()


def run_lm_eval(
    model: TraceModel,
    router: HardRouter,
    tasks: list[str],
    *,
    num_fewshot: int | None = None,
    limit: int | float | None = None,
) -> dict[str, Any]:
    """Execute lm-eval tasks and return its JSON-serializable result dictionary."""

    try:
        from lm_eval import simple_evaluate
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise LMEvalUnavailable("Install websleuth-models[eval]") from exc
    adapter = build_lm_eval_adapter(model, router)
    result = simple_evaluate(
        model=adapter,
        tasks=tasks,
        num_fewshot=num_fewshot,
        limit=limit,
        batch_size=1,
        log_samples=True,
    )
    return dict(result)
