import torch

from wm.model.cells import KnowledgeCell


def test_zero_initialized_cell_is_behavior_preserving(tiny_model):
    prompt = "Alpha"
    baseline = tiny_model.next_token_logits(prompt, cell_ids=[])
    cell = KnowledgeCell("c", tiny_model.target_module_names, tiny_model.hidden_size, rank=4)
    tiny_model.add_cell(cell)
    active = tiny_model.next_token_logits(prompt, cell_ids=["c"])
    assert torch.allclose(baseline, active, atol=0, rtol=0)


def test_cell_can_change_output(tiny_model):
    cell = KnowledgeCell("c", tiny_model.target_module_names, tiny_model.hidden_size, rank=4)
    tiny_model.add_cell(cell)
    for layer in cell.layers.values():
        torch.nn.init.normal_(layer.up.weight, std=0.1)
    baseline = tiny_model.next_token_logits("Alpha", cell_ids=[])
    active = tiny_model.next_token_logits("Alpha", cell_ids=["c"])
    assert not torch.allclose(baseline, active)


def test_old_fingerprint_excludes_new_cell(tiny_model):
    old = KnowledgeCell("old", tiny_model.target_module_names, tiny_model.hidden_size, rank=2)
    tiny_model.add_cell(old, trainable=False)
    before = tiny_model.state_fingerprint(include_cells=["old"])
    new = KnowledgeCell("new", tiny_model.target_module_names, tiny_model.hidden_size, rank=2)
    tiny_model.add_cell(new)
    with torch.no_grad():
        next(iter(new.parameters())).add_(1.0)
    after = tiny_model.state_fingerprint(include_cells=["old"])
    assert before == after
