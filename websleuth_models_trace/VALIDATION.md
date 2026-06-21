# Validation record

Validated locally on 2026-06-21 with the dependency-free tiny backend.

```bash
PYTHONPATH=src python -m compileall -q src scripts
PYTHONPATH=src pytest -q
PYTHONPATH=src python scripts/smoke_test.py
```

Results:

- Python byte-compilation completed without errors.
- 27 unit and integration tests passed.
- The smoke test crawled two documents, compiled a proof-backed episode, and generated five assimilation questions.

The included public benchmark and Hugging Face model adapters were not executed end-to-end in this environment because those runs require downloading the selected checkpoints and datasets. License-restricted benchmark adapters intentionally require user-provided local snapshots. AWS scheduler/account wiring is the only unimplemented infrastructure boundary.
