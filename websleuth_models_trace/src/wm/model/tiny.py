from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class ByteTokenizer:
    """Deterministic byte tokenizer for offline tests and synthetic experiments.

    The tokenizer is deliberately tiny and dependency-free. It is not intended as a
    production tokenizer; its purpose is to make every core Websleuth code path runnable
    on a laptop without downloading a checkpoint.
    """

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    pad_token = "<pad>"
    eos_token = "<eos>"
    vocab_size = 259

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [byte + 3 for byte in text.encode("utf-8", errors="replace")]
        if add_special_tokens:
            return [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, ids: Any, skip_special_tokens: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        values: list[int] = []
        for token in ids:
            token = int(token)
            if token < 3:
                if skip_special_tokens:
                    continue
                values.append(63)
            else:
                values.append(token - 3)
        return bytes(values).decode("utf-8", errors="replace")

    def __call__(
        self,
        text: str | list[str],
        *,
        return_tensors: str | None = None,
        truncation: bool = False,
        max_length: int | None = None,
        padding: bool | str = False,
        add_special_tokens: bool = True,
        **_: Any,
    ) -> dict[str, Any]:
        texts = [text] if isinstance(text, str) else list(text)
        rows = [self.encode(item, add_special_tokens=add_special_tokens) for item in texts]
        if truncation and max_length:
            rows = [row[-max_length:] for row in rows]
        width = max(len(row) for row in rows)
        if padding or len(rows) > 1:
            rows = [row + [self.pad_token_id] * (width - len(row)) for row in rows]
        masks = [[1 if token != self.pad_token_id else 0 for token in row] for row in rows]
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(rows, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }
        result_ids: Any = rows[0] if isinstance(text, str) else rows
        result_masks: Any = masks[0] if isinstance(text, str) else masks
        return {"input_ids": result_ids, "attention_mask": result_masks}


class TinyMLP(nn.Module):
    """Small residual MLP whose name is discoverable by KnowledgeCellBank."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(hidden)))


class TinyBlock(nn.Module):
    """Fast dependency-free residual block.

    The offline model intentionally avoids ``nn.MultiheadAttention``. Some CPU builds
    make one-token autoregressive generation through that kernel prohibitively slow.
    Production experiments use Hugging Face causal LMs; this block only exists to exercise
    cell allocation, routing, optimization, promotion, and serialization in CI.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.mlp = TinyMLP(hidden_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.mlp(self.ln(hidden))


class TinyCausalLM(nn.Module):
    """A very small causal-LM-compatible module for deterministic offline runs.

    It implements the subset of the Transformers causal-LM API used by Websleuth. The
    model is deliberately modest rather than competitive; meaningful experiments should
    pass ``model.backend: hf`` and an actual open-weight checkpoint.
    """

    def __init__(
        self,
        vocab_size: int = 259,
        hidden_size: int = 32,
        layers: int = 2,
        max_positions: int = 2048,
    ):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, vocab_size=vocab_size)
        self.max_positions = max_positions
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.pos = nn.Embedding(max_positions, hidden_size)
        self.layers = nn.ModuleList([TinyBlock(hidden_size) for _ in range(layers)])
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **_: Any,
    ) -> SimpleNamespace:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        _, length = input_ids.shape
        if length > self.max_positions:
            input_ids = input_ids[:, -self.max_positions :]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -self.max_positions :]
            if labels is not None:
                labels = labels[:, -self.max_positions :]
            length = input_ids.shape[1]
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0)
        hidden = self.embed(input_ids) + self.pos(positions)
        if attention_mask is not None:
            hidden = hidden * attention_mask.unsqueeze(-1).to(hidden.dtype)
        for layer in self.layers:
            hidden = layer(hidden)
        logits = self.lm_head(self.norm(hidden))
        loss = None
        if labels is not None and logits.shape[1] > 1:
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        elif labels is not None:
            loss = logits.sum() * 0.0
        return SimpleNamespace(logits=logits, loss=loss)

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 32,
        do_sample: bool = False,
        temperature: float = 1.0,
        pad_token_id: int | None = None,
        eos_token_id: int | None = None,
        **_: Any,
    ) -> torch.Tensor:
        del pad_token_id
        tokens = input_ids
        mask = attention_mask if attention_mask is not None else torch.ones_like(tokens)
        for _ in range(max_new_tokens):
            output = self.forward(tokens, attention_mask=mask)
            logits = output.logits[:, -1, :]
            if do_sample:
                probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
            mask = torch.cat([mask, torch.ones_like(next_token)], dim=1)
            if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                break
            if tokens.shape[1] >= self.max_positions:
                tokens = tokens[:, -self.max_positions :]
                mask = mask[:, -self.max_positions :]
        return tokens


def build_tiny_model(seed: int = 42) -> tuple[TinyCausalLM, ByteTokenizer]:
    # Large default CPU thread pools can make very small tensor operations slower by
    # two orders of magnitude. This setting affects only the dependency-free tiny backend.
    threads = max(1, int(os.environ.get("WEBSLEUTH_TINY_THREADS", "1")))
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(threads)
    except RuntimeError:
        # PyTorch only permits changing inter-op threads before parallel work begins.
        pass
    torch.manual_seed(seed)
    return TinyCausalLM(), ByteTokenizer()
