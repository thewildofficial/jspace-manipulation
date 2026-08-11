# Why we abandoned the 4B model

`Qwen/Qwen3.5-4B` did not pass the preregistered behavioral gate, so we stopped before performing any J-space readout or intervention on it.

Simplifying the task all the way to an explicitly supplied binary state did not eliminate the asymmetry. The model solved FLIP and A-input cases while failing the B/COPY identity case, particularly when semantic rule labels were removed. We therefore treated the failure as insufficient behavioral competence rather than continuing prompt optimization and moved to the next model with an official J-lens.

This is a model-selection result, not evidence that J-space lacks a separable reporting-policy representation.

## Decision rule

The pre-specified gate required at least 90% policy-following accuracy in **every** world-state × policy cell, together with positive margins. Overall accuracy was never allowed to hide a failed cell.

## Prompt-screen history

| Version | Change | Overall accuracy | Worst cell | Decision |
|---|---|---:|---:|---|
| v1 | Story fact; natural “same/other” objective | 70.8% | 0.0% | Answer-token and wording confounds |
| v2 | Symmetric COPY/FLIP and reward tables | 59.9% | 0.0% | Did not repair composition |
| v3 | Context-conditioned A/B token scoring | 59.9% | 0.0% | Confirmed v2 failure was not tokenization |
| v4 | Explicit `TRUE OPTION: A/B` added | 63.0% | 0.0% | Story inference was not the main bottleneck |
| v5 | Full symmetric function table, story retained | 93.2% | 70.8% | Strong aggregate, failed cell gate |
| v6 | Story-free toy function application | 84.4% | 0.0% | B/COPY failure sharpened without semantic labels |

In v6, named equations achieved 100% in three cells but only 75% for fact B + COPY, with a near-zero mean margin. The unlabeled R0/R1 matrix achieved 100% in three cells but 0% for fact B + identity row. The asymmetric pattern is inconsistent with a stable four-cell reporting function.

## Why stop rather than keep prompting?

Further prompt search would create three problems:

1. It would optimize against the same cells later used to claim behavioral competence.
2. It could produce a brittle prompt-specific trick rather than a mechanism that generalizes.
3. It would make any later J-space interpretation conditional on an undocumented prompt search.

The original research plan specified scaling the model when the smallest model lacked reliable behavior. The next candidate, `Qwen/Qwen3.6-27B`, also has an official 1,000-prompt Jacobian Lens and is therefore a cleaner fallback than fitting a new lens to an intermediate model.

## Files

- `raw/`: all six prompt-screen JSONL tables and resolved-run manifests.
- `summaries/`: four-cell bootstrap summaries for each version.
- `figures/`: behavioral-gate figures generated from the corresponding raw tables.

All versions use seed 1729 and direct conditional log-probability scoring. Version 3 onward records context-conditioned candidate token IDs (`A=32`, `B=33`) as a tokenizer sanity check.

