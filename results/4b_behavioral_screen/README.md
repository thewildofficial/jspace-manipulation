# Stage 1A — Gate 0: the 4B behavioral screen

## The question

Before looking inside a model, we asked whether it could perform the tiny task
we planned to study. The model was shown a fact, such as **A**, and a rule:

- **COPY / identity:** report the fact;
- **FLIP / swap:** report the other answer.

There were two possible facts and two possible rules. A model that cannot do
all four combinations reliably is not a good subject for a later internal
mechanism study.

## What happened

The `Qwen/Qwen3.5-4B` model did not pass the predeclared gate. The final
story-free toy version reached 84.4% overall, but one balanced cell was still
at 0%: fact **B** with the identity rule in the unlabeled R0/R1 wording.

| Prompt version | What changed | Overall | Weakest cell | Decision |
|---|---|---:|---:|---|
| P1 (file: v1) | Story fact and natural “same/other” wording | 70.8% | 0.0% | Too ambiguous |
| P2 (file: v2) | Symmetric COPY/FLIP and reward tables | 59.9% | 0.0% | Still failed |
| P3 (file: v3) | Context-conditioned A/B scoring | 59.9% | 0.0% | Not a tokenization fix |
| P4 (file: v4) | Explicit `TRUE OPTION: A/B` | 63.0% | 0.0% | Story was not the main issue |
| P5 (file: v5) | Full function table with story | 93.2% | 70.8% | Aggregate score hid a failed cell |
| P6 (file: v6) | Story-free toy function | 84.4% | 0.0% | Failure became clearer |
Prompting may have contributed, but repeated simplification did not remove the model’s asymmetric failure. We treated the 4B model as unsuitable for the later mechanism study, rather than claiming that the prompts were definitively perfect.

## Why this result matters

The failure is useful because it stopped us from blaming later J-space results
on a hidden lack of basic task ability. The model could solve some cases, but
not the full balanced function. We therefore scaled to the 27B model instead
of continuing to search for a prompt that might make the 4B model look good.

## What this does *not* show

It does not show that J-space lacks a reporting-policy representation. We never
ran a lens readout or intervention on this model. It is a model-selection
result, not a mechanistic result.

## The gate and the evidence

The gate required at least 90% accuracy in **every** fact × policy cell, with
positive score margins. Overall accuracy was not allowed to hide one failed
cell.

- [`raw/`](raw/) contains every version and its run manifest.
- [`summaries/`](summaries/) contains the four-cell summaries.
- [`figures/`](figures/) contains the corresponding plots.
- Prompt revisions are named **P1–P6** here; the raw filenames retain historical `v1`–`v6` labels. All use seed 1729 and direct conditional log-probability scoring.

The reusable prompt and scoring pieces are in [`../../src/jspace_policy/dataset.py`](../../src/jspace_policy/dataset.py) and [`../../scripts/analyze_results.py`](../../scripts/analyze_results.py).

[Back to the results map](../README.md) · [Back to the project README](../../README.md)
