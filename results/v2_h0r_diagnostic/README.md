# Stage 1E.3 — Protocol V2 recovery diagnosis (H0R-B)

## Why this stage existed

H0 failed in a specific way: a local edit sometimes worked, while repeating
the same edit across the selected band did not. H0R-B was an exploratory
diagnosis on the already-used (“burned”) country controls. It was allowed to
explain the failure and choose one candidate, but it could not validate that
candidate.

## What we tried

The diagnosis varied:

- individual layers from 28–52;
- where in the prompt the edit was applied;
- edit strength;
- different layer combinations, including odd/even sets;
- semantic, identity, random, unrelated, direct-answer, and out-of-window controls;
- a one-shot edit followed through later layers to see whether the model rebuilt its original state.

## What we learned

At full strength, single-layer edits moved answer scores in the desired
direction, but they also made large changes to the model’s output distribution.
A gentler strength of 0.5 found a much smaller, cleaner regime. The selected
seven-layer candidate produced:

- mean target-vs-source score gain **+4.34**;
- positive movement on **33/33** eligible burned country trials;
- mean output KL **0.227 nats**;
- mean residual-change ratio **0.017**;
- target top-1 on only **1/33** trials.

The controls were close to null: identity −0.010 and random-basis −0.011 mean
gain. The semantic effect was real and selective in this burned set, but the
low top-1 conversion was a warning. We froze the candidate anyway because the
fresh H0R-C test—not this exploratory result—was supposed to decide whether it
worked.

## The frozen candidate

The exact machine-readable protocol is
[`../../configs/v2/h0r_candidate_protocol.json`](../../configs/v2/h0r_candidate_protocol.json):

| Choice | Frozen value |
|---|---|
| Layers | 36, 37, 38, 39, 40, 41, 42 |
| Positions | Argument token through the end of the prompt |
| Operation | Two-coordinate pseudoinverse swap |
| Strength | α = 0.5 |
| Direction normalization | Unit L2 per layer |
| Hook point | Transformer block output residual |

No layer, position mask, strength, operation, normalization, validity limit,
or pass threshold changed during H0R-C.

## Certainty and claim boundary

**Certain:** this candidate was selected by the frozen diagnostic rule and
committed before fresh controls were opened.

**Not established:** that it generalizes to new arguments, preserves an
independent fact, or enables strategic reporting. H0R-C had to answer the
first of those questions, and it failed.

## Evidence files

- [`raw/`](raw/): all diagnostic JSONL rows.
- [`summaries/`](summaries/): layer, position, strength, reconstruction, and control summaries.
- [`figures/`](figures/): the nine diagnostic figures.
- [`run_manifest.json`](run_manifest.json): run metadata.
- [`../../docs/v2/h0r-diagnostic-report.md`](../../docs/v2/h0r-diagnostic-report.md): the technical interpretation.
- [`../v2_h0r_argument_validation_v2/README.md`](../v2_h0r_argument_validation_v2/README.md): the fresh prospective test.

The diagnostic runner is [`../../modal_h0r.py`](../../modal_h0r.py), the
measurements are implemented in [`../../src/jspace_policy/h0r_diagnostics.py`](../../src/jspace_policy/h0r_diagnostics.py),
and the summaries are rebuilt by [`../../scripts/analyze_h0r.py`](../../scripts/analyze_h0r.py).

[Previous gate: H0](../v2_smoke_tests/README.md) · [Next recovery phase: H0R-C](../v2_h0r_argument_validation_v2/README.md) · [Results map](../README.md)
