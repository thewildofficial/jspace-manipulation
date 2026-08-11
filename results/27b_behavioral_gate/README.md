# Qwen3.6-27B behavioral gate

`Qwen/Qwen3.6-27B` passed the Stage-1 toy behavioral gate and was selected for J-lens validation and causal experiments.

This is a competence result, not evidence for a latent reporting-policy mechanism. It establishes that failures in later readout or intervention experiments cannot be explained by the model being unable to apply the four-cell reporting function.

## Result

| Policy wording | Fact | Policy | Accuracy | Mean correct-answer margin | Weakest margin |
|---|---|---|---:|---:|---:|
| Named COPY/FLIP | A | COPY | 100% | 7.72 | 7.13 |
| Named COPY/FLIP | A | FLIP | 100% | 12.14 | 11.88 |
| Named COPY/FLIP | B | COPY | 100% | 9.12 | 8.88 |
| Named COPY/FLIP | B | FLIP | 100% | 7.56 | 7.25 |
| Unlabeled R0/R1 | A | identity | 100% | 6.33 | 6.13 |
| Unlabeled R0/R1 | A | swap | 100% | 4.02 | 3.25 |
| Unlabeled R0/R1 | B | identity | 100% | 2.56 | 2.25 |
| Unlabeled R0/R1 | B | swap | 100% | 4.50 | 4.13 |

Overall: **192/192 correct** across 24 base cases and both policy styles. Margins are natural-log probability differences between the expected and alternative one-token reports.

## Reproducibility metadata

- Model: `Qwen/Qwen3.6-27B`
- Resolved revision: `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`

## Why this resolves the 4B problem

The weakest 27B cell is the same unlabeled B/identity cell that the 4B model failed completely. The 27B model scores all 24 rows in that cell correctly, and its weakest individual margin is still +2.25. This makes model competence a controlled selection criterion rather than a post-hoc explanation for future mechanistic results.

## Files

- `raw/behavior_27b.jsonl`: one row per balanced prompt condition.
- `raw/behavior_27b.manifest.json`: resolved model revision, timing, GPU, and run ID.
- `summaries/behavior_27b_summary.csv`: scenario-bootstrap cell summaries.
- `figures/behavior_27b_gate.png`: cell-by-cell accuracy with uncertainty.
- `figures/paired_policy_shift_27b.png`: matched policy-induced answer shift by fact.
