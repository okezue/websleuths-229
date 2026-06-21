# Implementation status

This archive is a runnable research repository rather than a design-only scaffold.

## Implemented and exercised locally

- safe HTTP/file crawling, immutable content-addressed blobs, robots checks, URL canonicalization, redirect/size limits, and browser fallback boundary;
- HTML, text, PDF, image, and video parsing, including exact locators and optional OCR/transcription;
- SQLite evidence/proof store, FTS search, temporal claims, contradictions, source independence, and proof coverage;
- the typed WebREPL and all operations used by the evidence-collection loop;
- deterministic/local-model extraction and compilation into disjoint train/dev/test assimilation questions;
- frozen-backbone model wrapper, actual newly allocated residual cells, hard routing, cell serialization, capacity accounting, and exact state fingerprints;
- open-book/closed-book residual measurement, rank selection, cell optimization, promotion/rollback, route isolation, and parameter-invariance checks;
- iterative stream evaluation, native benchmark adapters, RULER-style context tests, code execution checks, provenance/security metrics, reports, plots, and scaling grids;
- dependency-free tiny backend and synthetic private micro-web for CI and offline systems tests;
- optional Hugging Face and `lm-eval` integrations for real model experiments.

The included test suite exercises the local core. Public benchmark execution naturally requires downloading the referenced datasets and model checkpoints. Benchmarks with redistribution or licensing constraints, including some LiveCodeBench, LegalBench, ToolAlpaca, and InfiniteBench snapshots, use fully implemented local JSON/JSONL adapters and fail explicitly when the user has not placed a snapshot under `data/`.

## Intentionally left as an infrastructure boundary

Only cloud scheduler wiring is a placeholder. `wm.infra.aws_placeholder` defines the submit/status/cancel contract but contains no account-specific AWS IAM, Batch, EKS, Slurm, S3, or networking assumptions. Local process execution is implemented in `wm.infra.local`.

## Scope caveats

- The structural non-forgetting guarantee is exact for immutable parameters and unchanged hard routes. Router errors remain a measurable behavioral failure mode rather than being hidden by the claim.
- The local Python code sandbox is best-effort. Public untrusted code evaluation should run inside a separately hardened container or VM.
- OCR, speech transcription, browser rendering, and public benchmark packages are optional extras; parsers retain raw source hashes even when those extras are unavailable.
- The tiny model is for systems validation, not scientific quality. Reported research results should use a real causal LM and multiple seeds.
