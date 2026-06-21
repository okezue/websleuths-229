import torch

from wm.analysis.cells import CellActivationTracer
from wm.model.cells import KnowledgeCell


def test_activation_tracer(tiny_model):
    cell = KnowledgeCell("c", tiny_model.target_module_names, tiny_model.hidden_size, rank=2)
    tiny_model.add_cell(cell)
    for layer in cell.layers.values():
        torch.nn.init.normal_(layer.up.weight, std=0.05)
    records = CellActivationTracer(tiny_model).trace(["alpha"], {"alpha": ["c"]})
    assert records
    assert records[0].calls > 0
