from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Iterable

import torch

from wm.config import AppConfig
from wm.core.hashing import stable_hash
from wm.core.schema import AssimilationReport, CellMetadata, ClaimRecord, EvidenceEpisode, QAItem
from wm.evidence.store import EvidenceStore
from wm.model.bank import CellBank
from wm.model.cells import KnowledgeCell
from wm.model.rank import GradientSpectrumRankEstimator, HiddenSpectrumRankEstimator, rank_from_residual
from wm.model.router import HardRouter
from wm.model.wrapper import TraceModel
from wm.reporting.aim_logger import AimLogger
from wm.train.losses import js_divergence
from wm.train.promotion import PromotionGate
from wm.train.residual import AssimilationResidualEstimator, exact_match

log = logging.getLogger(__name__)

_DEFAULT_OLD_PROBES = [
    "What is two plus two?",
    "Name the planet on which humans live.",
    "Write one sentence about photosynthesis.",
    "Return a JSON object with a boolean field named ok.",
    "Which ocean is the largest on Earth?",
]


def _ep_to_epoch(eid:str)->int:
    h=0
    for c in eid:h=(h*131+ord(c))&0x7FFFFFFF
    return h


@dataclass
class EvalSummary:
    nll: float
    accuracy: float
    n: int


class Assimilator:
    def __init__(
        self,
        cfg: AppConfig,
        model: TraceModel,
        router: HardRouter,
        bank: CellBank,
        store: EvidenceStore,
        aim_logger: AimLogger | None = None,
    ):
        self.cfg = cfg
        self.model = model
        self.router = router
        self.bank = bank
        self.store = store
        self.aim = aim_logger
        self.residual = AssimilationResidualEstimator(
            model,
            router,
            store,
            answer_template=cfg.training.answer_template,
        )
        self.rank_estimator = GradientSpectrumRankEstimator(
            cfg.cell.min_rank,
            cfg.cell.max_rank,
            cfg.cell.rank_energy,
        )
        self.hidden_rank_fallback = HiddenSpectrumRankEstimator(
            cfg.cell.min_rank,
            cfg.cell.max_rank,
            cfg.cell.rank_energy,
        )
        self.gate = PromotionGate(cfg.promotion)

    def _claims(self, episode: EvidenceEpisode) -> list[ClaimRecord]:
        return [claim for claim_id in episode.claim_ids if (claim := self.store.get_claim(claim_id))]

    def _route_metadata(self, episode: EvidenceEpisode, claims: list[ClaimRecord], rank: int) -> CellMetadata:
        entities = sorted({claim.subject for claim in claims} | {claim.object for claim in claims if len(claim.object) < 80})
        route_text = " ".join(
            [episode.domain, episode.topic]
            + [f"{claim.subject} {claim.predicate} {claim.object}" for claim in claims]
        )[:20_000]
        source_hashes = sorted(
            {
                doc.content_hash
                for doc_id in episode.document_ids
                if (doc := self.store.get_document(doc_id)) is not None
            }
        )
        valid_from = min((claim.valid_from for claim in claims if claim.valid_from), default=None)
        valid_to = max((claim.valid_to for claim in claims if claim.valid_to), default=None)
        cell_id = stable_hash({"episode": episode.episode_id, "rank": rank, "layers": self.model.target_module_names})
        probes = [qa.prompt for qa in episode.qa_items if qa.split == "test"][:20]
        return CellMetadata(
            cell_id=cell_id,
            episode_id=episode.episode_id,
            domain=episode.domain,
            topic=episode.topic,
            rank=rank,
            target_layers=self.model.target_module_names,
            route_text=route_text,
            entities=entities,
            valid_from=valid_from,
            valid_to=valid_to,
            source_hashes=source_hashes,
            claim_ids=episode.claim_ids,
            proof_span_ids=episode.span_ids,
            competence_probes=probes,
        )

    def _evaluate_qas(self, qas: Iterable[QAItem], extra_cells: Iterable[str] = ()) -> EvalSummary:
        items = list(qas)
        if not items:
            return EvalSummary(0.0, 0.0, 0)
        nll = accuracy = 0.0
        extra = list(extra_cells)
        for qa in items:
            prompt = self.residual.closed_prompt(qa)
            old = self.router.select(qa.prompt).cell_ids
            cell_ids = list(dict.fromkeys(old + extra))
            nll += self.model.score_answer(prompt, qa.answer, cell_ids=cell_ids).nll
            prediction = self.model.generate_text(prompt, cell_ids=cell_ids, max_new_tokens=48)
            accuracy += exact_match(prediction, qa.answer)
        return EvalSummary(nll / len(items), accuracy / len(items), len(items))

    def _temporary_router(self, metadata: CellMetadata) -> HardRouter:
        router = HardRouter(
            threshold=self.router.threshold,
            max_active=self.router.max_active,
            entity_bonus=self.router.entity_bonus,
        )
        for meta in self.router.metadata.values():
            router.metadata[meta.cell_id] = meta
        router.metadata[metadata.cell_id] = metadata
        router.rebuild()
        return router

    def _old_probe_invariants(self, provisional: HardRouter, new_cell_id: str) -> tuple[float, float, float, dict[str, object]]:
        probes = list(dict.fromkeys(self.bank.competence_probes() + _DEFAULT_OLD_PROBES))
        if not probes:
            return 0.0, 0.0, 0.0, {"n_probes": 0, "changed_routes": 0}
        false_routes = 0
        max_delta = 0.0
        changed_routes = 0
        for prompt in probes:
            before = self.router.select(prompt).cell_ids
            after = provisional.select(prompt).cell_ids
            if new_cell_id in after:
                false_routes += 1
            if before != [cell_id for cell_id in after if cell_id != new_cell_id]:
                changed_routes += 1
            if new_cell_id not in after:
                before_logits = self.model.next_token_logits(prompt, cell_ids=before)
                after_logits = self.model.next_token_logits(prompt, cell_ids=after)
                max_delta = max(max_delta, float((before_logits - after_logits).abs().max().cpu()))
        return (
            false_routes / len(probes),
            max_delta,
            changed_routes / len(probes),
            {"n_probes": len(probes), "changed_routes": changed_routes},
        )

    @staticmethod
    def _proof_coverage(qas: Iterable[QAItem]) -> float:
        items = list(qas)
        return sum(bool(item.proof_span_ids and item.claim_ids) for item in items) / max(len(items), 1)

    def assimilate(self, episode: EvidenceEpisode) -> AssimilationReport:
        if not self.cfg.training.enabled:
            return AssimilationReport(
                episode_id=episode.episode_id,
                residual={},
                rank=0,
                trained_steps=0,
                accepted=False,
                reason="training disabled",
            )
        train_qas = [qa for qa in episode.qa_items if qa.split == "train"]
        dev_qas = [qa for qa in episode.qa_items if qa.split == "dev"]
        test_qas = [qa for qa in episode.qa_items if qa.split == "test"]
        if not train_qas or not test_qas:
            return AssimilationReport(
                episode_id=episode.episode_id,
                residual={},
                rank=0,
                trained_steps=0,
                accepted=False,
                reason="episode lacks train or held-out questions",
            )
        residual_metrics = self.residual.evaluate(test_qas, generate=True)
        if self.aim:
            self.aim.track(residual_metrics.as_dict(),epoch=_ep_to_epoch(episode.episode_id),
                           context={'phase':'residual','episode':episode.episode_id,'domain':episode.domain})
        prompts = [self.residual.closed_prompt(qa) for qa in train_qas]
        rank_examples = [
            (self.residual.closed_prompt(qa), qa.answer, self.router.select(qa.prompt).cell_ids)
            for qa in train_qas
        ]
        try:
            rank, rank_diag = self.rank_estimator.estimate(self.model, rank_examples)
            rank_diag["gradient_estimator"] = 1.0
        except Exception as exc:
            log.warning("gradient rank estimator failed, trying hidden spectrum: %s", exc)
            try:
                rank, rank_diag = self.hidden_rank_fallback.estimate(self.model, prompts)
                rank_diag["hidden_fallback"] = 1.0
            except Exception as second_exc:
                log.warning("hidden rank estimator failed, using residual rule: %s", second_exc)
                rank = rank_from_residual(
                    residual_metrics.residual,
                    self.cfg.cell.min_rank,
                    self.cfg.cell.max_rank,
                )
                rank_diag = {"residual_fallback": 1.0}
        if self.aim:
            self.aim.track({'chosen_rank':float(rank)},epoch=_ep_to_epoch(episode.episode_id),
                           context={'phase':'rank','episode':episode.episode_id,'domain':episode.domain})
        claims = self._claims(episode)
        metadata = self._route_metadata(episode, claims, rank)
        old_ids = list(self.model.cells.keys())
        old_invariant = self.model.parameter_invariant_token(include_cells=old_ids)
        pre_test = self._evaluate_qas(test_qas)
        cell = KnowledgeCell(
            metadata.cell_id,
            self.model.target_module_names,
            self.model.hidden_size,
            rank,
            dropout=self.cfg.cell.dropout,
            scale=self.cfg.cell.scale,
        )
        if self.cfg.cell.initializer_path:
            from wm.model.initializer import CellInitializer

            CellInitializer.load_into(cell, self.cfg.cell.initializer_path, strict=False)
        self.model.add_cell(cell, trainable=True)
        optimizer = torch.optim.AdamW(
            cell.parameters(),
            lr=self.cfg.training.learning_rate,
            weight_decay=self.cfg.training.weight_decay,
        )
        history: list[dict[str, float]] = []
        best_dev = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        stale = 0
        steps = 0
        for step in range(self.cfg.training.max_steps):
            qa = train_qas[step % len(train_qas)]
            closed_prompt = self.residual.closed_prompt(qa)
            open_prompt = self.residual.open_prompt(qa)
            old_route = self.router.select(qa.prompt).cell_ids
            student_cells = list(dict.fromkeys(old_route + [metadata.cell_id]))
            supervised = self.model.answer_loss(closed_prompt, qa.answer, cell_ids=student_cells)
            distill = torch.tensor(0.0, device=self.model.device)
            if self.cfg.training.distill_weight > 0 and qa.proof_span_ids:
                student_logits = self.model.answer_logits(closed_prompt, qa.answer, cell_ids=student_cells)
                with torch.no_grad():
                    teacher_logits = self.model.answer_logits(open_prompt, qa.answer, cell_ids=old_route)
                distill = js_divergence(student_logits, teacher_logits)
            loss = self.cfg.training.supervised_weight * supervised + self.cfg.training.distill_weight * distill
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cell.parameters(), self.cfg.training.grad_clip)
            optimizer.step()
            steps += 1
            step_metrics={
                "step": float(steps),
                "loss": float(loss.detach().cpu()),
                "supervised": float(supervised.detach().cpu()),
                "distill": float(distill.detach().cpu()),
            }
            history.append(step_metrics)
            if self.aim:
                self.aim.track({'loss':step_metrics['loss'],'supervised':step_metrics['supervised'],
                                'distill':step_metrics['distill']},step=steps,
                               epoch=_ep_to_epoch(episode.episode_id),
                               context={'phase':'train','episode':episode.episode_id,
                                        'cell':metadata.cell_id,'domain':episode.domain})
            if steps % self.cfg.training.eval_every == 0 or steps == self.cfg.training.max_steps:
                dev = self._evaluate_qas(dev_qas or train_qas, extra_cells=[metadata.cell_id])
                history[-1]["dev_nll"] = dev.nll
                history[-1]["dev_accuracy"] = dev.accuracy
                if self.aim:
                    self.aim.track({'dev_nll':float(dev.nll),'dev_accuracy':float(dev.accuracy)},step=steps,
                                   epoch=_ep_to_epoch(episode.episode_id),
                                   context={'phase':'dev','episode':episode.episode_id,'domain':episode.domain})
                if dev.nll + 1e-6 < best_dev:
                    best_dev = dev.nll
                    best_state = {name: tensor.detach().cpu().clone() for name, tensor in cell.state_dict().items()}
                    stale = 0
                else:
                    stale += 1
                if stale >= self.cfg.training.early_stop_patience:
                    break
        if best_state is not None:
            cell.load_state_dict(best_state)
        post_test = self._evaluate_qas(test_qas, extra_cells=[metadata.cell_id])
        test_gain = post_test.accuracy - pre_test.accuracy
        nll_gain = pre_test.nll - post_test.nll
        parameter_invariant = self.model.parameters_unchanged(old_invariant, include_cells=old_ids)
        cell_hash = self.model.cell_fingerprint(metadata.cell_id)
        metadata = metadata.model_copy(
            update={
                "parameter_count": cell.parameter_count,
                "parameter_hash": cell_hash,
                "metrics": {
                    "pre_test_nll": pre_test.nll,
                    "post_test_nll": post_test.nll,
                    "pre_test_accuracy": pre_test.accuracy,
                    "post_test_accuracy": post_test.accuracy,
                    "test_gain": test_gain,
                    **{f"rank_{key}": float(value) for key, value in rank_diag.items()},
                },
            }
        )
        provisional = self._temporary_router(metadata)
        route_fp, max_logit_delta, route_churn, route_diag = self._old_probe_invariants(provisional, metadata.cell_id)
        proof_coverage = self._proof_coverage(episode.qa_items)
        if self.aim:
            self.aim.track({'route_fp':float(route_fp),'max_logit_delta':float(max_logit_delta),
                            'route_churn':float(route_churn),'proof_coverage':float(proof_coverage),
                            'parameter_invariant':1.0 if parameter_invariant else 0.0},
                           epoch=_ep_to_epoch(episode.episode_id),
                           context={'phase':'invariants','episode':episode.episode_id,'domain':episode.domain})
        decision = self.gate.decide(
            test_gain=test_gain,
            test_accuracy=post_test.accuracy,
            nll_gain=nll_gain,
            proof_coverage=proof_coverage,
            route_false_positive=route_fp,
            max_old_logit_delta=max_logit_delta,
            parameter_invariant=parameter_invariant,
            old_route_churn=route_churn,
        )
        if decision.accepted:
            self.model.freeze_cell(metadata.cell_id)
            metadata = metadata.model_copy(update={"promoted": True})
            self.router.add(metadata)
            self.bank.save(cell, metadata)
            cell_id: str | None = metadata.cell_id
        else:
            self.model.remove_cell(metadata.cell_id)
            cell_id = None
        metrics = {
            "pre_test_nll": pre_test.nll,
            "post_test_nll": post_test.nll,
            "pre_test_accuracy": pre_test.accuracy,
            "post_test_accuracy": post_test.accuracy,
            "test_gain": test_gain,
            "nll_gain": nll_gain,
            "proof_coverage": proof_coverage,
            "route_false_positive": route_fp,
            "max_old_logit_delta": max_logit_delta,
            "old_route_churn": route_churn,
            "parameter_count": float(cell.parameter_count),
        }
        if self.aim:
            self.aim.track({**metrics,'accepted':1.0 if decision.accepted else 0.0,'rank':float(rank),
                            'trained_steps':float(steps)},
                           epoch=_ep_to_epoch(episode.episode_id),
                           context={'phase':'episode_summary','episode':episode.episode_id,
                                    'domain':episode.domain,'cell':metadata.cell_id})
        return AssimilationReport(
            episode_id=episode.episode_id,
            residual=residual_metrics.as_dict(),
            rank=rank,
            trained_steps=steps,
            accepted=decision.accepted,
            reason=decision.reason,
            cell_id=cell_id,
            metrics=metrics,
            invariants={
                "parameter_invariant": parameter_invariant,
                "checks": decision.checks,
                **route_diag,
            },
            history=history,
        )
