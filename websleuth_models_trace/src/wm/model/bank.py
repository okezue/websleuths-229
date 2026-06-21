from __future__ import annotations

from pathlib import Path
from typing import Iterable

from safetensors.torch import load_file, save_file

from wm.core.io import ensure_dir, read_json, write_json
from wm.core.schema import CellMetadata
from wm.model.cells import KnowledgeCell


class CellBank:
    def __init__(self, root: str | Path):
        self.root = ensure_dir(root)
        self.manifest_path = self.root / "manifest.json"
        if not self.manifest_path.exists():
            write_json(self.manifest_path, {"cells": []})

    def _dir(self, cell_id: str) -> Path:
        return self.root / cell_id

    def save(self, cell: KnowledgeCell, metadata: CellMetadata) -> CellMetadata:
        cell_dir = ensure_dir(self._dir(cell.cell_id))
        parameter_hash = metadata.parameter_hash
        updated = metadata.model_copy(
            update={
                "parameter_count": cell.parameter_count,
                "parameter_hash": parameter_hash,
            }
        )
        state = {name: tensor.detach().cpu().contiguous() for name, tensor in cell.state_dict().items()}
        save_file(state, str(cell_dir / "cell.safetensors"))
        write_json(cell_dir / "metadata.json", updated.model_dump(mode="json"))
        manifest = read_json(self.manifest_path)
        cells = [cid for cid in manifest.get("cells", []) if cid != cell.cell_id] + [cell.cell_id]
        write_json(self.manifest_path, {"cells": sorted(cells)})
        return updated

    def metadata(self, cell_id: str) -> CellMetadata:
        return CellMetadata.model_validate(read_json(self._dir(cell_id) / "metadata.json"))

    def list_metadata(self) -> list[CellMetadata]:
        manifest = read_json(self.manifest_path)
        return [self.metadata(cell_id) for cell_id in manifest.get("cells", []) if (self._dir(cell_id) / "metadata.json").exists()]

    def load(self, cell_id: str, hidden_size: int, *, dropout: float = 0.0, scale: float = 1.0) -> tuple[KnowledgeCell, CellMetadata]:
        metadata = self.metadata(cell_id)
        cell = KnowledgeCell(
            cell_id=cell_id,
            target_layers=metadata.target_layers,
            hidden_size=hidden_size,
            rank=metadata.rank,
            dropout=dropout,
            scale=scale,
        )
        state = load_file(str(self._dir(cell_id) / "cell.safetensors"))
        cell.load_state_dict(state)
        cell.freeze()
        return cell, metadata

    def load_all(self, hidden_size: int, *, dropout: float = 0.0, scale: float = 1.0) -> list[tuple[KnowledgeCell, CellMetadata]]:
        return [self.load(meta.cell_id, hidden_size, dropout=dropout, scale=scale) for meta in self.list_metadata()]

    def competence_probes(self) -> list[str]:
        return [probe for meta in self.list_metadata() for probe in meta.competence_probes]

    def source_hashes(self) -> set[str]:
        return {digest for meta in self.list_metadata() for digest in meta.source_hashes}
