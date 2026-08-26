# Stage 1E.2 — Protocol V2 causal gate (H0)

## The question

The first V2 test asked whether the planned J-space edit could work on a
separate positive-control task before we tried anything about strategic
reporting.

The control task came from held-out country facts. The intervention tried to
swap one answer for another at different layers and positions. The key
comparison was:

- **one layer / final position:** a small local edit;
- **one layer / all positions:** the same layer, but across the prompt;
- **band / final position:** several middle layers at the final position;
- **band / all positions:** the preregistered reference topology.

## What happened

The required reference topology failed. It made **0/90** baseline-eligible
trials produce the target answer as top-1.

| Edit | Target top-1 | Target score movement | Plain-language reading |
|---|---:|---:|---|
| One layer / final position | 1/90 (1.1%) | +0.82 | Usually too weak |
| One layer / all positions | 25/90 (27.8%) | +5.10 | A surprisingly useful local edit |
| Band / final position | 0/90 (0.0%) | +0.22 | Too weak |
| Band / all positions | 0/90 (0.0%) | +0.58 | Required topology failed |

The one-layer/all-position condition worked especially well on countries:
24/33 eligible country trials reached target top-1. That was the most
counterintuitive result in the run. Repeating the same symmetric edit through
the whole band did not strengthen it; it erased or disrupted the useful local
effect.

## Why the gate stopped

The preregistered reference needed at least 25% target top-1 and at least a
10-point advantage over the one-layer/final-position comparison. It achieved
0% and did not beat that comparison. Because this was the required causal
instrument check, H1–H8 and every strategic-reporting experiment remained
closed.

The score movement is still informative: the edit was not simply numerically
broken. But “the score moved” is weaker than “the requested answer became the
most likely answer,” so it could not validate the instrument.

## What was checked

- Numerical transport and readout parity passed.
- An analytically known coordinate swap passed.
- The finite-direction sign check passed.
- Tokenization exclusions and baseline failures were recorded rather than hidden.
- Downstream reconstruction and independent fact preservation were **not** measured here and must not be inferred.

## Evidence and reproducibility

- [`raw/flexible_generalization.jsonl`](raw/flexible_generalization.jsonl): the intervention rows.
- [`raw/h0_summary.json`](raw/h0_summary.json): the gate summary.
- [`run_manifest.json`](run_manifest.json): run, model, code, and config metadata.
- [`summaries/topology_summary.csv`](summaries/topology_summary.csv): regenerated topology results.
- [`figures/topology_effect.png`](figures/topology_effect.png): target conversion and score movement.
- [`../v2_workspace_mapping/README.md`](../v2_workspace_mapping/README.md): the separate layer-selection artifact.
- [`../../docs/v2/final-report.md`](../../docs/v2/final-report.md): the formal report.

The GPU runner is [`../../modal_v2.py`](../../modal_v2.py), the intervention
math is in [`../../src/jspace_policy/interventions.py`](../../src/jspace_policy/interventions.py),
and the deterministic rebuild is [`../../scripts/analyze_v2_h0.py`](../../scripts/analyze_v2_h0.py).

[Previous run: observational pilot](../27b_observational_pilot/README.md) · [Next recovery phase: H0R-B diagnosis](../v2_h0r_diagnostic/README.md) · [Results map](../README.md)
