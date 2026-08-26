# Stage 1B — Gate 0 scale-up: the 27B behavioral gate

## The question

Could the larger `Qwen/Qwen3.6-27B` model reliably apply the same small rule?
This is a scale-up within Stage 1, not a new scientific stage.
This is a competence check. It tells us whether later failures might simply
come from the model not understanding the task.

## What happened

It passed every one of the 192 tested rows.

| Prompt wording | Fact | Rule | Result |
|---|---|---|---:|
| Named COPY/FLIP | A | COPY | 24/24 |
| Named COPY/FLIP | A | FLIP | 24/24 |
| Named COPY/FLIP | B | COPY | 24/24 |
| Named COPY/FLIP | B | FLIP | 24/24 |
| Unlabeled R0/R1 | A | identity | 24/24 |
| Unlabeled R0/R1 | A | swap | 24/24 |
| Unlabeled R0/R1 | B | identity | 24/24 |
| Unlabeled R0/R1 | B | swap | 24/24 |

The weakest individual answer margin was still +2.25 in natural-log units.
In plainer terms, the correct answer was not merely first by a hair.

## Why it matters

The 27B model solved the exact cell that defeated the 4B model: fact B with
the unlabeled identity rule. That makes the model switch a planned selection
decision rather than an after-the-fact explanation for a later result.

This result does not show that the model has separate fact and policy
representations. It only shows that it can perform the visible toy task.

## Reproducibility

- Model: `Qwen/Qwen3.6-27B`
- Resolved model revision: `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`
- [`raw/behavior_27b.jsonl`](raw/behavior_27b.jsonl): one row per balanced prompt condition.
- [`raw/behavior_27b.manifest.json`](raw/behavior_27b.manifest.json): model, timing, GPU, and run metadata.
- [`summaries/behavior_27b_summary.csv`](summaries/behavior_27b_summary.csv): cell summaries.
- [`figures/behavior_27b_gate.png`](figures/behavior_27b_gate.png): cell-by-cell accuracy.
- [`figures/paired_policy_shift_27b.png`](figures/paired_policy_shift_27b.png): matched policy-induced answer shift.

The run is assembled by [`../../modal_app.py`](../../modal_app.py), using the
shared prompt and analysis code in [`../../src/jspace_policy/`](../../src/jspace_policy/README.md).

[Previous run: the 4B screen](../4b_behavioral_screen/README.md) · [Next run: lens integrity](../27b_lens_integrity/README.md) · [Results map](../README.md)
