from __future__ import annotations

import contextlib
import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn

from wm.config import ModelConfig
from wm.model.cells import KnowledgeCell


@dataclass
class AnswerScore:
    nll: float
    token_count: int
    normalized_probability: float


def _tensor_from_output(output: Any) -> tuple[torch.Tensor | None, Any]:
    if isinstance(output, torch.Tensor):
        return output, None
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0], output
    return None, output


def _replace_output(output: Any, tensor: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return tensor
    if isinstance(output, tuple):
        return (tensor, *output[1:])
    return output


def _module_hidden_size(module: nn.Module, fallback: int | None = None) -> int:
    for child in module.modules():
        if isinstance(child, nn.Linear):
            return int(child.out_features)
    if fallback:
        return fallback
    raise ValueError(f"cannot infer hidden size for {type(module).__name__}")


class TraceModel(nn.Module):
    """Frozen causal LM plus append-only residual cells injected through forward hooks."""

    def __init__(
        self,
        base_model: nn.Module,
        tokenizer: Any | None,
        *,
        target_module_patterns: list[str] | None = None,
        target_last_fraction: float = 0.25,
        max_input_tokens: int = 2048,
        max_new_tokens: int = 96,
    ):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.cells = nn.ModuleDict()
        self.active_cell_ids: list[str] = []
        self.active_gates: dict[str, float] = {}
        self._hook_handles: list[Any] = []
        self._target_module_patterns = target_module_patterns or [".mlp", ".feed_forward", ".ffn"]
        self._target_module_names = self._discover_targets(target_last_fraction)
        self._target_modules = dict(self.base_model.named_modules())
        self.hidden_size = int(
            getattr(getattr(base_model, "config", None), "hidden_size", 0)
            or _module_hidden_size(self._target_modules[self._target_module_names[0]])
        )
        self.freeze_base()
        self._install_hooks()

    @classmethod
    def from_pretrained(cls, cfg: ModelConfig) -> "TraceModel":
        if cfg.name in {"__tiny__", "tiny", "wm/tiny"}:
            from wm.model.tiny import build_tiny_model

            model, tokenizer = build_tiny_model()
            return cls(
                model,
                tokenizer,
                target_module_patterns=cfg.target_module_patterns,
                target_last_fraction=cfg.target_last_fraction,
                max_input_tokens=cfg.max_input_tokens,
                max_new_tokens=cfg.max_new_tokens,
            )
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install websleuth-models[models]") from exc
        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[cfg.dtype]
        tokenizer = AutoTokenizer.from_pretrained(cfg.name, trust_remote_code=cfg.trust_remote_code)
        if getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        kwargs: dict[str, Any] = {"trust_remote_code": cfg.trust_remote_code, "torch_dtype": dtype}
        if cfg.device == "auto":
            kwargs["device_map"] = "auto" if torch.cuda.is_available() else None
        model = AutoModelForCausalLM.from_pretrained(cfg.name, **kwargs)
        if cfg.device not in {"auto", ""}:
            model.to(cfg.device)
        return cls(
            model,
            tokenizer,
            target_module_patterns=cfg.target_module_patterns,
            target_last_fraction=cfg.target_last_fraction,
            max_input_tokens=cfg.max_input_tokens,
            max_new_tokens=cfg.max_new_tokens,
        )

    @property
    def device(self) -> torch.device:
        return next(self.base_model.parameters()).device

    @property
    def target_module_names(self) -> list[str]:
        return list(self._target_module_names)

    def _discover_targets(self, fraction: float) -> list[str]:
        candidates: list[str] = []
        for name, module in self.base_model.named_modules():
            if not name:
                continue
            if any(name.endswith(pattern) for pattern in self._target_module_patterns):
                if any(isinstance(child, nn.Linear) for child in module.modules()):
                    candidates.append(name)
        if not candidates:
            # Fallback to repeated blocks with linears, excluding the output head.
            for name, module in self.base_model.named_modules():
                if name and "lm_head" not in name and sum(isinstance(child, nn.Linear) for child in module.children()) >= 2:
                    candidates.append(name)
        if not candidates:
            raise ValueError("no compatible transformer MLP modules found")
        count = max(1, int(round(len(candidates) * max(0.0, min(fraction, 1.0)))))
        return candidates[-count:]

    def _install_hooks(self) -> None:
        modules = dict(self.base_model.named_modules())
        for name in self._target_module_names:
            module = modules[name]

            def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any, module_name: str = name) -> Any:
                hidden, original = _tensor_from_output(output)
                if hidden is None or not self.active_cell_ids:
                    return output
                result = hidden
                for cell_id in self.active_cell_ids:
                    if cell_id not in self.cells:
                        continue
                    cell = self.cells[cell_id]
                    if not cell.applies_to(module_name):
                        continue
                    gate = float(self.active_gates.get(cell_id, 1.0))
                    if gate:
                        result = result + gate * cell.residual(module_name, hidden)
                return _replace_output(original if original is not None else output, result)

            self._hook_handles.append(module.register_forward_hook(hook))

    def freeze_base(self) -> None:
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        self.base_model.eval()

    def add_cell(self, cell: KnowledgeCell, *, trainable: bool = True) -> None:
        if cell.cell_id in self.cells:
            raise ValueError(f"cell already exists: {cell.cell_id}")
        cell.to(self.device)
        self.cells[cell.cell_id] = cell
        if trainable:
            cell.unfreeze()
        else:
            cell.freeze()

    def remove_cell(self, cell_id: str) -> None:
        if cell_id in self.cells:
            del self.cells[cell_id]
        self.active_cell_ids = [cid for cid in self.active_cell_ids if cid != cell_id]
        self.active_gates.pop(cell_id, None)

    def freeze_cell(self, cell_id: str) -> None:
        self.cells[cell_id].freeze()

    @contextlib.contextmanager
    def activated(self, cell_ids: Iterable[str], gates: dict[str, float] | None = None) -> Iterator[None]:
        previous_ids = list(self.active_cell_ids)
        previous_gates = dict(self.active_gates)
        self.active_cell_ids = [cid for cid in cell_ids if cid in self.cells]
        self.active_gates = {cid: float((gates or {}).get(cid, 1.0)) for cid in self.active_cell_ids}
        try:
            yield
        finally:
            self.active_cell_ids = previous_ids
            self.active_gates = previous_gates

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.base_model(*args, **kwargs)

    def _encode(self, text: str) -> dict[str, torch.Tensor]:
        if self.tokenizer is None:
            raise RuntimeError("tokenizer is required")
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        return {key: value.to(self.device) for key, value in encoded.items() if isinstance(value, torch.Tensor)}

    def generate_text(
        self,
        prompt: str,
        *,
        cell_ids: Iterable[str] = (),
        max_new_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> str:
        encoded = self._encode(prompt)
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": getattr(self.tokenizer, "pad_token_id", None),
        }
        if temperature > 0:
            kwargs["temperature"] = temperature
        with self.activated(cell_ids), torch.no_grad():
            output = self.base_model.generate(**encoded, **kwargs)
        new_tokens = output[0, encoded["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _prompt_answer_tokens(self, prompt: str, answer: str) -> tuple[dict[str, torch.Tensor], int]:
        if self.tokenizer is None:
            raise RuntimeError("tokenizer is required")
        prompt_ids = list(self.tokenizer(prompt, add_special_tokens=True)["input_ids"])
        answer_ids = list(self.tokenizer(answer, add_special_tokens=False)["input_ids"])
        if len(answer_ids) >= self.max_input_tokens:
            answer_ids = answer_ids[: max(1, self.max_input_tokens - 1)]
        eos = getattr(self.tokenizer, "eos_token_id", None)
        # A tokenizer may append EOS when special tokens are requested. Remove it so the
        # answer is actually conditioned on the prompt rather than placed after termination.
        if prompt_ids and eos is not None and prompt_ids[-1] == eos:
            prompt_ids = prompt_ids[:-1]
        available = max(1, self.max_input_tokens - len(answer_ids))
        if len(prompt_ids) > available:
            prompt_ids = prompt_ids[-available:]
        full_ids = prompt_ids + answer_ids
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}, len(prompt_ids)

    def answer_logits(self, prompt: str, answer: str, *, cell_ids: Iterable[str] = ()) -> torch.Tensor:
        encoded, prompt_len = self._prompt_answer_tokens(prompt, answer)
        with self.activated(cell_ids):
            output = self.base_model(**encoded)
        logits = output.logits[:, :-1, :]
        start = max(prompt_len - 1, 0)
        return logits[:, start:, :]

    def answer_loss(self, prompt: str, answer: str, *, cell_ids: Iterable[str] = ()) -> torch.Tensor:
        encoded, prompt_len = self._prompt_answer_tokens(prompt, answer)
        labels = encoded["input_ids"].clone()
        labels[:, :prompt_len] = -100
        with self.activated(cell_ids):
            output = self.base_model(**encoded, labels=labels)
        if hasattr(output, "loss") and output.loss is not None:
            return output.loss
        logits = output.logits[:, :-1, :]
        targets = labels[:, 1:]
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)

    def score_answer(self, prompt: str, answer: str, *, cell_ids: Iterable[str] = ()) -> AnswerScore:
        with torch.no_grad():
            loss = self.answer_loss(prompt, answer, cell_ids=cell_ids)
        token_count = max(1, len(self.tokenizer(answer, add_special_tokens=False)["input_ids"]))
        nll = float(loss.detach().cpu())
        return AnswerScore(nll=nll, token_count=token_count, normalized_probability=float(torch.exp(-loss).detach().cpu()))

    def next_token_logits(self, prompt: str, *, cell_ids: Iterable[str] = ()) -> torch.Tensor:
        encoded = self._encode(prompt)
        with self.activated(cell_ids), torch.no_grad():
            output = self.base_model(**encoded)
        return output.logits[:, -1, :].detach()

    def parameter_invariant_token(self, *, include_cells: Iterable[str] | None = None) -> dict[str, tuple[int, int, tuple[int, ...], str]]:
        """Capture a zero-copy token proving that existing parameters were not mutated.

        PyTorch increments a tensor's internal version counter on every in-place write.
        Recording that counter together with the storage pointer, shape, and dtype avoids
        copying a multi-billion-parameter backbone to CPU after every web episode. The
        optimizer is separately constructed only from the new cell, so this token is the
        scalable promotion-time invariant. ``state_fingerprint`` remains available for
        slower content-addressed audits and checkpoint manifests.
        """

        token: dict[str, tuple[int, int, tuple[int, ...], str]] = {}
        for name, parameter in self.base_model.named_parameters():
            token[f"base:{name}"] = (
                int(parameter.data_ptr()),
                int(parameter._version),
                tuple(parameter.shape),
                str(parameter.dtype),
            )
        ids = sorted(include_cells if include_cells is not None else self.cells.keys())
        for cell_id in ids:
            if cell_id not in self.cells:
                continue
            for name, parameter in self.cells[cell_id].named_parameters():
                token[f"cell:{cell_id}:{name}"] = (
                    int(parameter.data_ptr()),
                    int(parameter._version),
                    tuple(parameter.shape),
                    str(parameter.dtype),
                )
        return token

    def parameters_unchanged(
        self,
        token: dict[str, tuple[int, int, tuple[int, ...], str]],
        *,
        include_cells: Iterable[str] | None = None,
    ) -> bool:
        return self.parameter_invariant_token(include_cells=include_cells) == token

    def state_fingerprint(self, *, include_cells: Iterable[str] | None = None) -> str:
        h = hashlib.sha256()
        for name, tensor in sorted(self.base_model.state_dict().items()):
            h.update(name.encode())
            h.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        ids = sorted(include_cells if include_cells is not None else self.cells.keys())
        for cell_id in ids:
            if cell_id not in self.cells:
                continue
            for name, tensor in sorted(self.cells[cell_id].state_dict().items()):
                h.update(f"{cell_id}:{name}".encode())
                h.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

    def cell_fingerprint(self, cell_id: str) -> str:
        h = hashlib.sha256()
        for name, tensor in sorted(self.cells[cell_id].state_dict().items()):
            h.update(name.encode())
            h.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for cell in self.cells.values() for parameter in cell.parameters() if parameter.requires_grad]

    def close(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
