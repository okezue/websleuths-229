# Retention Experiment Summary

This document summarizes the completed retention experiment from `results/retention/retention_meta-llama_Llama-3.2-1B.json` and `results/retention/retention_meta-llama_Llama-3.2-1B_checkpoint.json`.

## What This Experiment Does

This experiment tests whether dreaming reduces forgetting during sequential topic adaptation. It replaces the older ablation setup with a cleaner retention study: train on several topics one after another, and after every update, reevaluate the model on all earlier topics using held-out data. That gives a direct measurement of how much earlier knowledge drops after later training.

The setup was:

- model: `meta-llama/Llama-3.2-1B`
- adaptation: LoRA
- conditions: `naive` and `dreaming`
- seeds: `42`, `43`, `44`
- steps: `200` per topic
- batch size: `2`
- topics:
  - `forensics`
  - `chemistry`
  - `finance`
  - `legal`
  - `medicine`

The two conditions are simple:

- `naive`: train only on the current topic
- `dreaming`: train on the current topic while also regularizing against the pre-update teacher using anchors and prompts from earlier topic training rows

Each topic was split into train and held-out eval data, and the training size was balanced across topics so one domain would not dominate the sequence. The final sizes were:

| Topic      | Train Rows | Eval Rows |
| ---------- | ---------- | --------- |
| Forensics  | 105        | 26        |
| Chemistry  | 105        | 116       |
| Finance    | 105        | 33        |
| Legal      | 105        | 97        |
| Medicine   | 105        | 189       |

## Main Results

The main result is straightforward: dreaming helped. Across the three seeds, mean forgetting dropped from `0.0133 ± 0.0031` under naive training to `0.0068 ± 0.0016` with dreaming, which is about a `49%` reduction. Final average held-out accuracy also improved slightly, from `0.4627 ± 0.0018` to `0.4718 ± 0.0008`, so the gain did not come from simply freezing the model and avoiding new learning. The strongest effect showed up in the stability metrics: final anchor NLL fell from `4.0779 ± 0.0422` to `3.6767 ± 0.0123`, and drift to the base model dropped from `4.4167 ± 0.2933` to `0.4933 ± 0.0415`.

| Metric                 | Naive            | Dreaming         | Delta     |
| ---------------------- | ---------------- | ---------------- | --------- |
| Mean forgetting        | 0.0133 ± 0.0031  | 0.0068 ± 0.0016  | -49.1%    |
| Final avg held-out acc | 0.4627 ± 0.0018  | 0.4718 ± 0.0008  | +0.0091   |
| Final anchor NLL       | 4.0779 ± 0.0422  | 3.6767 ± 0.0123  | -0.4012   |
| Final drift to base    | 4.4167 ± 0.2933  | 0.4933 ± 0.0415  | -3.9234   |

The pattern was also consistent at the seed level. Dreaming beat naive on overall forgetting in all three seeds, and it also produced better final average held-out accuracy in all three seeds:

| Seed | Naive Forgetting | Dreaming Forgetting | Naive Final Avg Acc | Dreaming Final Avg Acc |
| ---- | ---------------- | ------------------- | ------------------- | ---------------------- |
| 42   | 0.0153           | 0.0076              | 0.4611              | 0.4719                 |
| 43   | 0.0097           | 0.0049              | 0.4646              | 0.4726                 |
| 44   | 0.0148           | 0.0078              | 0.4625              | 0.4709                 |

Looking at topic-level forgetting, the biggest gains are on forensics, chemistry, and especially legal. Finance shows only a small improvement and is clearly noisier. Medicine is omitted from the table below because it is the last topic in the sequence, so there are no later topics after it that could cause forgetting.

| Topic      | Naive  | Dreaming | Reduction |
| ---------- | ------ | -------- | --------- |
| Forensics  | 0.0150 | 0.0084   | -44.3%    |
| Chemistry  | 0.0135 | 0.0077   | -43.1%    |
| Finance    | 0.0072 | 0.0069   | -4.4%     |
| Legal      | 0.0174 | 0.0041   | -76.7%    |

```text
Mean Forgetting
Naive      0.0133 | █████████████
Dreaming   0.0068 | ███████

Final Drift To Base
Naive      4.4167 | ████████████████████████████████████████
Dreaming   0.4933 | ████

Final Avg Held-Out Accuracy
Naive      0.4627 | ██████████████████████████████████████████████
Dreaming   0.4718 | ███████████████████████████████████████████████
```

## What These Results Mean

The cleanest reading is that dreaming reduces forgetting and dramatically reduces drift during sequential topic adaptation. This experiment is much better aligned with the actual claim than the older ablations because it measures held-out topic performance after later updates instead of mostly measuring fit on training-derived data. It is also not a story where dreaming blocks learning: final held-out accuracy is slightly better with dreaming, while stability is much better.

The most important takeaways are:

- overall forgetting was cut by about half
- this held across all 3 seeds
- final held-out accuracy was slightly better with dreaming
- drift to the base model was dramatically lower with dreaming
- the strongest topic-level gains were on forensics, chemistry, and legal
- finance was the weakest and noisiest topic

There are still a few limits worth keeping in mind. The evaluation here is next-token prediction on held-out text chunks, not a QA benchmark or a downstream reasoning benchmark. Eval set sizes also differ by topic, which likely explains some of the finance noise. Still, those caveats do not change the main result. In this five-topic, three-seed retention study, dreaming reduced measured forgetting by about 49 percent, improved final held-out accuracy slightly, and kept the model much closer to its base behavior throughout the sequence.
