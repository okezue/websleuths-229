from datetime import UTC, datetime

from wm.core.schema import CellMetadata
from wm.model.router import HardRouter


def meta(cell_id, text, entities, valid_from=None, valid_to=None):
    return CellMetadata(cell_id=cell_id, episode_id="e", domain="d", topic="t", rank=2, target_layers=["x"], route_text=text, entities=entities, valid_from=valid_from, valid_to=valid_to)


def test_hard_router_no_route_default_and_entities():
    router = HardRouter(threshold=0.2, max_active=1, entity_bonus=0.5)
    router.add(meta("c1", "Aurelia debt equity finance", ["Aurelia Holdings"]))
    assert router.select("weather tomorrow").cell_ids == []
    assert router.select("What is Aurelia Holdings debt ratio?").cell_ids == ["c1"]


def test_temporal_routing():
    router = HardRouter(threshold=0.0, max_active=2)
    router.add(meta("old", "office holder", ["Office"], valid_to=datetime(2030, 1, 1, tzinfo=UTC)))
    router.add(meta("new", "office holder", ["Office"], valid_from=datetime(2030, 1, 1, tzinfo=UTC)))
    assert router.select("Office holder", at_time=datetime(2029, 1, 1, tzinfo=UTC)).cell_ids == ["old"]
    assert router.select("Office holder", at_time=datetime(2031, 1, 1, tzinfo=UTC)).cell_ids == ["new"]
