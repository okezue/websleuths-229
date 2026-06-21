# Websleuth Models: TRACE-Web

TRACE-Web is a complete research implementation of **proof-carrying web continual learning**. It treats web retrieval as an executable evidence automaton and compiles only the model's verified open-book/closed-book residual into **append-only neural knowledge cells**. The backbone and all previously promoted cells remain frozen.

The repository is designed to test four claims:

1. Web evidence can be surfaced as typed, queryable, provenance-preserving objects rather than API-generated prose.
2. A model can learn durable new facts and procedures from that evidence at inference time.
3. Improvements can accumulate across domains without overwriting prior parameters.
4. Capacity growth, assimilation, retention, routing, context scaling, and general-knowledge regression can be measured in one iterative protocol.

## What is implemented

- Model-native crawler with robots handling, safe URL policy, immutable blob storage, HTML/PDF/image/video parsing, optional JavaScript rendering, downsampling, OCR, and speech transcription.
- SQLite evidence store with exact locators, content hashes, FTS search, temporal validity, contradiction links, proof graphs, and episode manifests.
- A typed WebREPL with `seek`, `open`, `inspect`, `extract`, `assert`, `challenge`, `link`, `ask`, and `commit` operations.
- Deterministic and local-model claim extractors. No external extraction API is required.
- Open-book/closed-book assimilation-residual estimation.
- Actual capacity growth through newly allocated residual MLP cells inserted into selected transformer MLP layers.
- Hard sparse routing with temporal validity and a no-route default.
- Cell training, source-addressable checkpoints, promotion gates, exact state-fingerprint checks, and rollback.
- Frozen, RAG, shared-cell, replay, EATRD, DPMU, EAB-SSC, and append-only TRACE baselines, plus a matched baseline-matrix runner.
- Iterative continual-learning evaluation with domain-gain matrices, retention, forgetting, backward transfer, general-knowledge regression, provenance, routing, security, and capacity metrics.
- Direct benchmark adapters plus an optional `lm-eval` bridge.
- Synthetic micro-web generation, including HTML, PDF, image, table, contradiction, temporal-update, and prompt-injection pages.
- RULER-style long-context tests, including needle retrieval, multi-needle retrieval, variable tracking, and aggregation.
- Local execution is fully implemented. AWS-specific launch plumbing is intentionally isolated in `wm.infra.aws_placeholder`.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[models,dev]'

# Generate a private, contamination-free web stream.
websleuth micro-web --output runs/micro_web --episodes 12

# Crawl and compile its evidence.
websleuth crawl file://$PWD/runs/micro_web/index.html --config configs/quick.yaml
websleuth compile-evidence --topic "micro web" --domain synthetic --config configs/quick.yaml

# Run a stream with any Hugging Face causal LM.
websleuth stream --config configs/stream_350m.yaml --manifest runs/micro_web/manifest.json

# Compare acquisition, retention, regression, and parameter cost across baselines.
websleuth baseline-matrix --config configs/stream_350m.yaml --manifest runs/micro_web/manifest.json \
  --method frozen --method rag --method shared --method replay --method eatrd \
  --method dpmu --method eab_ssc --method trace

# List all benchmark definitions.
websleuth benchmark-catalog
```

## Iterative evaluation

At every learning step, the stream runner records:

- current-domain pre/post gain;
- all previously learned-domain accuracies;
- mean and worst-case forgetting;
- retention area under the curve;
- backward and forward transfer;
- MMLU, MMLU-Pro, ARC-Challenge, HellaSwag, Winogrande, TruthfulQA, GSM8K, BBH, and IFEval-style regression checks;
- FinQA, LexGLUE CaseHOLD, ChemBench/SciKnowEval, MedQA/PubMedQA, GPQA, HumanEval/MBPP/LiveCodeBench-style domain measures;
- LongBench, RULER, InfiniteBench/SCROLLS-style long-context measures;
- route false positives/negatives and route-set churn;
- evidence support precision and complete-proof coverage;
- added parameters, rank, train FLOPs proxy, latency, and gain per million added parameters.

See [docs/benchmarks.md](docs/benchmarks.md) and [docs/experiments.md](docs/experiments.md).

## Structural non-forgetting claim

TRACE-Web never updates the backbone or a promoted cell. A newly learned cell changes an old input only if the hard router elects to activate it. The repository therefore measures two separate quantities:

- **parameter invariance:** hashes of the backbone and old cells are identical before and after an update;
- **behavioral isolation:** old probes preserve the same route set, avoid false activation of the new cell, and retain identical logits within a configured numerical tolerance.

This is more precise than claiming that a regularizer “completely prevents forgetting.” Routing mistakes are still possible and are measured explicitly.

## Repository layout

```text
src/wm/web          crawler, parsers, rendering, media extraction
src/wm/evidence     schemas, store, extraction, verification, graph compiler
src/wm/repl         executable evidence VM
src/wm/model        knowledge cells, hard router, bank, injected model
src/wm/train        residual measurement, QA generation, assimilation, promotion
src/wm/eval         benchmark adapters, continual protocol, context/security tests
src/wm/synthetic    private micro-web and stream generation
src/wm/baselines    frozen, RAG, shared-cell, replay baselines
src/wm/reporting    JSON/CSV/plot/HTML reports
src/wm/infra        local runner and AWS integration boundary
```

## Data and access policy

The crawler is built for public material available through normal HTTP or browser execution. It does not bypass authentication, paywalls, DRM, robots restrictions, or access controls. Private-network addresses are blocked by default. Web content is treated as untrusted data and never as an instruction to the crawler or learner.

## Reproducibility

All generated episodes, fetched bytes, parsers, evidence spans, claims, queries, cell metadata, routes, and evaluations are content-addressed or seeded. Every promoted cell stores the exact source hashes and held-out probes that justified promotion.
