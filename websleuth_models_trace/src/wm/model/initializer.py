from __future__ import annotations

from pathlib import Path

from safetensors.torch import load_file, save_file

from wm.model.cells import KnowledgeCell


class CellInitializer:
    """Serializable meta-learned initialization for a fixed cell shape."""

    @staticmethod
    def save(cell: KnowledgeCell, path: str | Path) -> None:
        state = {name: tensor.detach().cpu().contiguous() for name, tensor in cell.state_dict().items()}
        save_file(state, str(path))

    @staticmethod
    def load_into(cell: KnowledgeCell, path: str | Path, strict: bool = True) -> None:
        state = load_file(str(path))
        missing, unexpected = cell.load_state_dict(state, strict=False)
        if strict and (missing or unexpected):
            raise ValueError(f"initializer shape mismatch; missing={missing}, unexpected={unexpected}")
