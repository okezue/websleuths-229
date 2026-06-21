# Contributing

1. Create an isolated environment with Python 3.11 or newer.
2. Install `pip install -e '.[dev,models]'`.
3. Run `pytest`, `python -m compileall -q src scripts`, and `ruff check src tests scripts`.
4. Add a deterministic unit test for every change to evidence semantics, routing, promotion, or evaluation.
5. Never make a generated summary the sole support for an assimilated claim. New extractors must preserve source span identifiers and locators.
6. Never mutate the backbone or a promoted cell in an append-only experiment. Baselines that do so must be named and isolated under `wm.baselines`.
7. New benchmarks belong in `configs/benchmarks.yaml`, with a loader, metric, source/reference, fixed sampling seed, and an explicit unavailable-data failure mode.
