from __future__ import annotations

from dataclasses import dataclass

from wm.model.wrapper import TraceModel


@dataclass
class CapacityMetrics:
    backbone_parameters: int
    cell_parameters: int
    total_parameters: int
    active_cell_parameters: int
    n_cells: int
    n_active_cells: int


def capacity_metrics(model: TraceModel, active_cell_ids: list[str] | None = None) -> CapacityMetrics:
    active = active_cell_ids or []
    backbone = sum(parameter.numel() for parameter in model.base_model.parameters())
    cells = sum(cell.parameter_count for cell in model.cells.values())
    active_params = sum(model.cells[cell_id].parameter_count for cell_id in active if cell_id in model.cells)
    return CapacityMetrics(backbone, cells, backbone + cells, active_params, len(model.cells), len(active))
