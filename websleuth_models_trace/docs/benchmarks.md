# Benchmark plan

The benchmark registry is executable and lives in `configs/benchmarks.yaml`. Dataset connectors use Hugging Face `datasets`, local JSONL files, or synthetic generators. The runner evaluates a fixed sample list at every continual-learning step so changes are paired rather than confounded by resampling.

## Domain learning suites

### Finance

- **FinQA**: numerical reasoning over financial reports and tables.
- **TAT-QA**: table-and-text financial QA.
- **ConvFinQA**: conversational financial reasoning.
- Optional local additions: FinanceBench, SEC filing change sets, and private post-cutoff earnings questions.

Primary metrics: numerical exact match, relative numeric tolerance, program/result consistency, and evidence-span support.

### Law

- **LexGLUE CaseHOLD**: select the correct legal holding.
- **LexGLUE SCOTUS/ECtHR**: legal topic and outcome classification.
- **LegalBench**: diverse legal reasoning tasks.
- Private temporal sets: changes in statutes, regulations, and precedents with explicit valid dates.

Primary metrics: multiple-choice accuracy, macro-F1 for classification, temporal correctness, and supporting authority recovery.

### Chemistry and science

- **ChemBench**: broad chemistry knowledge and reasoning.
- **SciKnowEval**: multi-level scientific knowledge in chemistry, physics, biology, and materials science.
- **GPQA Diamond**: graduate-level science reasoning.
- MMLU college chemistry and related subject subsets are also reported separately.

### Medicine

- **MedQA**: USMLE-style medical questions.
- **PubMedQA**: evidence-based biomedical QA.
- **MedMCQA**: medical entrance-exam questions.
- Private temporal sets: new trial results, label changes, and guideline updates, clearly separated from medical advice.

### Code and tool use

- **HumanEval** and **MBPP**: executable code generation.
- **LiveCodeBench**: contamination-aware, time-indexed coding problems when a licensed/local snapshot is available.
- **ToolAlpaca**: map instructions and API descriptions to tool calls.

Execution is sandboxed. `pass@1` is available in the core runner; higher `pass@k` uses repeated generations.

## General knowledge and regression suite

The default regression panel includes:

- **MMLU** for broad subject knowledge;
- **MMLU-Pro** for harder multi-task knowledge and reasoning;
- **ARC-Challenge** for science reasoning;
- **HellaSwag** for commonsense completion;
- **WinoGrande** for coreference and commonsense;
- **TruthfulQA MC1** for truthfulness;
- **GSM8K** for grade-school mathematical reasoning;
- **BIG-Bench Hard** for diverse difficult reasoning;
- **IFEval** for precise instruction following;
- **CommonsenseQA**, **PIQA**, and **OpenBookQA** as optional expansions.

The central continual-learning plot reports each general benchmark at baseline and after every domain episode. The aggregate score is never used alone: per-benchmark changes and confidence intervals are retained.

## Long-context and memory suite

### Synthetic RULER-style tests

Implemented without external data:

- single needle / pass-key retrieval;
- multiple needles;
- UUID and number retrieval;
- variable tracking;
- aggregation and counting;
- distractor and prompt-injection robustness.

Run at approximately 2K, 4K, 8K, 16K, 32K, 64K, and 128K tokens, subject to the tested model's native limit. Report exact-match, prefill latency, decode latency, and bytes/tokens processed.

### Public long-context suites

- **LongBench** for QA, summarization, retrieval, few-shot, and code tasks;
- **SCROLLS** for long-document language understanding;
- **InfiniteBench** for very long contexts when a local snapshot is present;
- optional NarrativeQA, Qasper, MuSiQue, HotpotQA, GovReport, QMSum, MultiNews, LCC, and RepoBench-P subsets through LongBench.

## Continual metrics

For domain `d` after update `t`, let `A[d,t]` be held-out performance. The report includes:

- immediate gain `A[d,t]-A[d,t-1]`;
- retained gain relative to the original baseline;
- maximum historical score and forgetting from that maximum;
- mean accuracy over all learned domains;
- count of domains retaining statistically positive gain;
- general benchmark delta and Pareto frontier between acquisition and regression.

## Provenance, routing, and safety tests

Every benchmark answer can optionally require a proof. Separate suites measure:

- complete proof recovery;
- citation precision;
- contradiction recognition;
- outdated versus current fact routing;
- false activation of new cells on old prompts;
- prompt injection, malicious instructions in documents, SSRF attempts, and oversized content handling.
