# Architecture

## 1. Memory hierarchy

TRACE-Web separates three kinds of memory:

1. **Frozen backbone**: pretrained language and reasoning capability.
2. **Append-only knowledge cells**: durable, compact assimilation of stable evidence.
3. **Evidence store**: exact, volatile, low-confidence, high-precision, or legally sensitive material.

At step `t`, the model is

\[
F_t(x)=F_\theta(x)+\sum_{j\in R_t(x)} C_j(x),
\]

where `F_theta` and every previously promoted `C_j` are immutable. `R_t` is a hard sparse router with a no-route default.

## 2. Web evidence automaton

The WebREPL exposes a typed instruction set:

```text
seek(query, constraints)             -> document handles
open(document)                       -> document record
inspect(document, selector, channel) -> evidence spans
extract(spans)                       -> candidate claims
assert(spans, proposition)           -> supported claim
challenge(claim)                     -> supporting and opposing claims
link(claims, relation)               -> proof-graph edge
ask(question)                        -> answer and proof
commit(claims)                       -> immutable evidence episode
```

Every assertion points to immutable content hashes and exact locators: DOM paths and character ranges for HTML, pages and boxes for PDFs, regions for images, and time ranges for video/audio.

## 3. Assimilation residual

For a verified evidence episode `E`, the question builder creates direct, inverse, cloze, temporal, negative-control, and compositional queries. The same model is evaluated:

- **closed book**: question only;
- **open book**: question plus proof-producing evidence spans.

The residual combines answer-NLL and exact-answer improvement:

\[
R(E)=\mathbb{E}_{q\sim Q(E)} w(q,E)
[\ell_{closed}(q)-\ell_{open}(q)].
\]

Only reliable, stable, useful episodes with high residual allocate capacity.

## 4. Knowledge cells

A cell consists of newly allocated residual MLPs in selected transformer layers:

\[
C_j(h)=U_j\,\sigma(V_j h).
\]

The `up` matrices are zero-initialized, so a newly created cell is initially behavior preserving. The cell rank is selected from the hidden-state spectrum of the episode prompts, constrained by configured minimum and maximum ranks.

Cells are stored independently with:

- rank and target layers;
- route prototypes and entities;
- temporal validity;
- source hashes and proof IDs;
- promotion metrics;
- competence probes;
- exact parameter fingerprint.

## 5. Structural isolation

Promotion checks:

- held-out closed-book gain;
- proof coverage;
- base and old-cell fingerprints unchanged;
- old probes preserve their route set;
- non-routed old logits remain within numerical tolerance;
- router false-positive rate;
- contradiction and temporal tests.

The exact guarantee is conditional: if an old input receives the same route set after a new cell is added, its computation uses the same parameters and therefore should remain numerically identical up to deterministic kernel tolerance.

## 6. Temporal updates

New facts do not overwrite old facts. Cells and claims have `valid_from` and `valid_to`. A superseding claim closes the previous interval and creates a new interval. Historical and current questions intentionally route differently.

## 7. Web safety

- private network ranges are blocked by default;
- robots policies are honored;
- bytes are bounded and content-addressed;
- rendering runs without credentials;
- page text is always untrusted data;
- prompt-injection text cannot call tools directly;
- authentication, paywall, DRM, and access-control bypasses are out of scope.


## Data-dependent capacity growth

Before allocating a cell, TRACE inserts a temporary zero-effect rank-1 probe cell and collects held-out loss gradients at the selected MLP outputs. It chooses the smallest bottleneck rank whose singular spectrum explains the configured energy fraction. This makes growth depend on the dimensionality of the evidence residual rather than a fixed per-topic adapter size. A hidden-state spectrum and a scalar residual rule are retained as explicit fallbacks.
