# Retention Experiment Summary

This document summarizes the two completed retention experiments from `results/retention/retention_meta-llama_Llama-3.2-1B.json` and `results/retention/retention_Qwen_Qwen2.5-3B.json`.

## What This Experiment Does

These experiments test whether dreaming reduces forgetting during sequential topic adaptation. The setup is the same in both runs: train on several topics one after another, and after every update, reevaluate the model on all earlier topics using held-out data. That gives a direct measurement of how much earlier knowledge drops after later training.

The shared setup was:

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

The protocol was the same in both runs, but the frozen topic snapshots were not exactly identical. The balanced train sizes and eval sizes were:

| Model      | Train Rows Per Topic | Eval Rows by Topic (`F / C / Fi / L / M`) |
| ---------- | -------------------- | ----------------------------------------- |
| Llama 1B   | 105                  | 26 / 116 / 33 / 97 / 189                  |
| Qwen 3B    | 140                  | 40 / 101 / 35 / 161 / 207                 |

That means the two runs should be read mainly as two replications of the same effect, not as a perfectly apples-to-apples model ranking.

## Main Results

The main result is very consistent: dreaming helped on both models. In both runs it reduced forgetting, improved final held-out accuracy slightly, and cut stability drift by a large margin. The exact numbers differ, but the direction of the result is the same all the way through.

| Model    | Mean Forgetting                              | Final Avg Held-Out Acc                        | Final Anchor NLL                          | Final Drift To Base                         |
| -------- | -------------------------------------------- | --------------------------------------------- | ----------------------------------------- | ------------------------------------------- |
| Llama 1B | `0.0133 ± 0.0031 -> 0.0068 ± 0.0016` `-49.1%` | `0.4627 ± 0.0018 -> 0.4718 ± 0.0008` `+0.0091` | `4.0779 ± 0.0422 -> 3.6767 ± 0.0123`      | `4.4167 ± 0.2933 -> 0.4933 ± 0.0415` `-88.8%` |
| Qwen 3B  | `0.0119 ± 0.0004 -> 0.0044 ± 0.0003` `-62.9%` | `0.5111 ± 0.0005 -> 0.5145 ± 0.0006` `+0.0034` | `3.1730 ± 0.0576 -> 2.8221 ± 0.0145`      | `2.1892 ± 0.2678 -> 0.1051 ± 0.0405` `-95.2%` |

The same consistency shows up at the seed level. Dreaming beat naive on all three seeds for both models on every headline metric:

| Model    | Better Forgetting | Better Final Avg Acc | Lower Anchor NLL | Lower Drift |
| -------- | ----------------- | -------------------- | ---------------- | ----------- |
| Llama 1B | 3 / 3             | 3 / 3                | 3 / 3            | 3 / 3       |
| Qwen 3B  | 3 / 3             | 3 / 3                | 3 / 3            | 3 / 3       |

Across both runs together, that is `6 / 6` wins for dreaming on forgetting, `6 / 6` on final held-out accuracy, `6 / 6` on anchor NLL, and `6 / 6` on drift.

The topic-level pattern is also similar across both models. Legal, chemistry, and forensics show the strongest gains. Finance improves too, but it is the weakest and noisiest topic in both runs. Medicine is omitted from the table below because it is the last topic in the sequence, so there are no later topics after it that could cause forgetting.

| Topic      | Llama Reduction | Qwen Reduction | Shared Read |
| ---------- | --------------- | -------------- | ----------- |
| Forensics  | -44.3%          | -62.1%         | strong improvement |
| Chemistry  | -43.1%          | -63.7%         | strong improvement |
| Finance    | -4.4%           | -29.2%         | weakest and noisiest |
| Legal      | -76.7%          | -82.9%         | strongest improvement |

There is also a common sequence-level pattern. In both runs, naive training stayed below the base-model average held-out accuracy until the final topic. Dreaming crossed earlier and stayed ahead sooner. On Llama that crossover happened after `legal`. On Qwen it happened after `chemistry`.

```text
Mean Forgetting Reduction
Llama 1B   49.1% | ██████████
Qwen 3B    62.9% | █████████████

Drift Reduction To Base
Llama 1B   88.8% | ██████████████████
Qwen 3B    95.2% | ███████████████████

Final Avg Held-Out Accuracy Gain
Llama 1B   +0.0091 | █████████
Qwen 3B    +0.0034 | ███

Shared Topic Pattern
Legal      strongest gain in both runs
Finance    weakest / noisiest in both runs
```

## What These Results Mean

The cleanest reading is that the retention result replicated. Dreaming was not just better on one model, or one seed, or one metric. It worked on both Llama 1B and Qwen 3B, and it improved the same things in both places: lower forgetting, lower drift, and slightly better final held-out performance.

That is important because it makes the result much harder to dismiss as a one-off. The two models are different, the data snapshots are not identical, and the absolute numbers differ, but the qualitative outcome is the same. In that sense, the shared conclusion is stronger than either single run by itself.

There is also a useful nuance in both runs. Dreaming is not just “freezing” the model. Final held-out accuracy is still slightly better with dreaming on both models. At the same time, dreaming keeps the model much closer to its base behavior. So the result is not stability at the cost of learning. It is better stability with no obvious learning collapse.

The most important takeaways are:

- dreaming reduced forgetting on both models
- dreaming improved final held-out accuracy on both models
- dreaming dramatically reduced drift on both models
- the result held across all `6` seed-model combinations
- legal, chemistry, and forensics were the strongest common gains
- finance was the weakest and noisiest topic in both runs

There are still a few limits worth keeping in mind. The evaluation here is next-token prediction on held-out text chunks, not a QA benchmark or a downstream reasoning benchmark. The frozen topic snapshots also differed between the two runs, so cross-model comparisons should be treated as qualitative rather than exact. Still, those caveats do not change the main result. Across two model families, dreaming consistently reduced measured forgetting, slightly improved final held-out accuracy, and kept the model much closer to its base behavior throughout the sequence.
