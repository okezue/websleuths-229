from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from wm.core.schema import EvidenceEpisode, QAItem
from wm.evidence.store import EvidenceStore
from wm.model.router import HardRouter
from wm.model.wrapper import TraceModel


@dataclass
class ResidualMetrics:
    closed_nll: float
    open_nll: float
    nll_gain: float
    closed_exact: float
    open_exact: float
    exact_gain: float
    residual: float
    n: int

    def as_dict(self) -> dict[str, float]:
        return self.__dict__.copy()


def normalize_answer(text: str) -> str:
    return " ".join(text.strip().lower().split()).strip(" .,:;!?\"'")


def exact_match(prediction: str, answer: str) -> float:
    pred = normalize_answer(prediction)
    gold = normalize_answer(answer)
    return float(pred == gold or (gold and gold in pred))


class AssimilationResidualEstimator:
    def __init__(
        self,
        model: TraceModel,
        router: HardRouter,
        store: EvidenceStore,
        answer_template: str = "Question: {question}\nAnswer:",
        max_evidence_chars: int = 6000,
    ):
        self.model = model
        self.router = router
        self.store = store
        self.answer_template = answer_template
        self.max_evidence_chars = max_evidence_chars

    def closed_prompt(self, qa: QAItem) -> str:
        return self.answer_template.format(question=qa.prompt) + " "

    def evidence_text(self, qa: QAItem) -> str:
        chunks: list[str] = []
        total = 0
        for span_id in qa.proof_span_ids:
            span = self.store.get_span(span_id)
            if not span:
                continue
            snippet = span.text.strip()
            if total + len(snippet) > self.max_evidence_chars:
                snippet = snippet[: max(0, self.max_evidence_chars - total)]
            if snippet:
                chunks.append(f"[{span_id}] {snippet}")
                total += len(snippet)
            if total >= self.max_evidence_chars:
                break
        return "\n".join(chunks)

    def open_prompt(self, qa: QAItem) -> str:
        evidence = self.evidence_text(qa)
        return f"Verified evidence:\n{evidence}\n\n{self.closed_prompt(qa)}"

    def evaluate(
        self,
        qas: Iterable[QAItem],
        *,
        extra_closed_cells: Iterable[str] = (),
        generate: bool = True,
    ) -> ResidualMetrics:
        items = list(qas)
        if not items:
            return ResidualMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
        closed_nll = open_nll = closed_exact = open_exact = 0.0
        extra = list(extra_closed_cells)
        for qa in items:
            route = self.router.select(qa.prompt).cell_ids
            closed_ids = list(dict.fromkeys(route + extra))
            closed_prompt = self.closed_prompt(qa)
            open_prompt = self.open_prompt(qa)
            closed_nll += self.model.score_answer(closed_prompt, qa.answer, cell_ids=closed_ids).nll
            open_nll += self.model.score_answer(open_prompt, qa.answer, cell_ids=route).nll
            if generate:
                closed_pred = self.model.generate_text(closed_prompt, cell_ids=closed_ids, max_new_tokens=48)
                open_pred = self.model.generate_text(open_prompt, cell_ids=route, max_new_tokens=48)
                closed_exact += exact_match(closed_pred, qa.answer)
                open_exact += exact_match(open_pred, qa.answer)
        n = len(items)
        c_nll = closed_nll / n
        o_nll = open_nll / n
        c_exact = closed_exact / n if generate else 0.0
        o_exact = open_exact / n if generate else 0.0
        nll_gain = c_nll - o_nll
        exact_gain = o_exact - c_exact
        residual = max(0.0, nll_gain) + max(0.0, exact_gain)
        return ResidualMetrics(c_nll, o_nll, nll_gain, c_exact, o_exact, exact_gain, residual, n)
