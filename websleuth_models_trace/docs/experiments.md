# Experimental protocol

## Primary hypothesis

A proof-carrying web learner can accumulate domain improvements through append-only capacity while preserving prior parameters, prior routes, and broad general capability.

## Stage 1: private micro-web

Generate contamination-free facts and procedures with random entities and values. Each episode contains:

- two independent supporting pages;
- one plausible contradiction;
- one irrelevant decoy;
- one prompt-injection page;
- at least two modalities among HTML, PDF, table, image, and video;
- train/dev/test queries split by graph edge and template family.

Recommended grid:

| Axis | Values |
|---|---|
| Backbone | 125M, 350M, 1B, 3B |
| Episodes | 32, 128, 512, 2,048 |
| Added capacity | 0.05%, 0.2%, 0.5%, 2% of base |
| Seeds | 5 below 1B, 3 at 1B and 3B |
| Modalities | text, table, PDF, image, video |
| Episode type | new fact, procedure, contradiction, temporal update, composition |

At each step:

1. evaluate new episode closed book;
2. execute WebREPL under fixed request/byte/time budget;
3. compile and verify a proof graph;
4. measure open-book and closed-book performance;
5. compute residual and choose whether to grow;
6. train only the new cell;
7. remove evidence and evaluate held-out closed-book queries;
8. re-evaluate every prior episode;
9. run routing, state invariance, general regression, and security checks;
10. promote or discard the cell.

## Stage 2: real temporal streams

Use frozen snapshots and explicit publication-date splits. Include public post-cutoff events plus private nonce facts embedded in realistic pages. The private facts are the strongest contamination control.

## Stage 3: multimodal evidence

Test information visible only in:

- a chart value;
- image text;
- a diagram relation;
- a video event or subtitle;
- a text-image contradiction;
- a PDF table cell.

## Stage 4: long stream

Run at least 1,000 episodes with recurring entities, superseding facts, low-value details, repeated concepts, cross-domain composition, and deliberate source disagreement.

## Baselines

- frozen model;
- frozen model with RAG/WebREPL;
- one shared residual cell updated sequentially;
- shared cell with replay;
- one fixed isolated cell per episode;
- fixed-rank append-only cells;
- append-only cells without the residual trigger;
- TRACE-Web with adaptive rank, proof residual, and hard routing;
- current EATRD/DPMU/EAB-SSC implementations when ported into the same evidence and evaluation substrate;
- external TTT/context-distillation baselines where licenses and implementations permit.

## Metrics

### Assimilation

\[
\Delta_{assim}=A_{closed,post}-A_{closed,pre}
\]

and fraction of open-book gain compiled:

\[
\eta_{compile}=\frac{A_{closed,post}-A_{closed,pre}}
{A_{open}-A_{closed,pre}}.
\]

### Continual learning

- mean and worst-case forgetting;
- retention AUC;
- backward transfer;
- forward transfer;
- cumulative domain gain;
- number of domains simultaneously above their pre-learning baselines.

### General regression

Track absolute and relative deltas on broad knowledge, reasoning, truthfulness, instruction-following, and commonsense suites. Improvement is allowed and reported; the objective is not merely “no change.”

### Structural invariance

- exact base and old-cell state fingerprints;
- old route-set equality;
- max and mean non-routed logit difference;
- deterministic generation equality.

### Evidence

- support precision;
- complete-proof coverage;
- independent-source count;
- contradiction detection;
- temporal correctness;
- citation recovery after cell promotion.

### Efficiency and scaling

- requests, bytes, and evidence tokens per assimilated fact;
- added parameters per retained fact;
- gain per million added parameters;
- cell-bank size and active cells per query;
- adaptation and inference latency;
- context-length accuracy and latency curves.


## Matched baseline matrix

`websleuth baseline-matrix` (or `scripts/run_baseline_matrix.py`) rebuilds the same evidence stream from the same model initialization for frozen, RAG, one shared mutable cell, replay, EATRD, DPMU, EAB-SSC, and TRACE. Each method receives an isolated database and cell directory. The report records episode-level acquisition, the full prior-domain retention matrix, general-benchmark deltas, and final parameter cost. This avoids comparing methods that saw different web snapshots or question samples.
