# V5-RBG-2 — semantic action-outcome override

## The question

Can the model report what an action will cause while failing to choose that
action when the action is a meaningful assertion?

The game used an explicit contrarian response map. Meaningful claims were
compared with opaque tokens so that the action itself, rather than the causal
rule, was the main difference.

## What happened

The model reported explicit option consequences at 97.66% accuracy. It chose
opaque actions correctly at 100%, but chose meaningful contrarian claims
correctly only 57.81% of the time. The 42.19-point gap passed the frozen test.

In the strongest error subset, the model had reported both consequences
correctly, then selected the claim whose words matched the desired receiver
response even though the contrarian rule said that claim would cause the other
response. This is a behavioral knowledge/action dissociation. It is not yet a
mechanistic explanation.

## Read next

- [`FINDINGS.md`](FINDINGS.md) — the frozen result and controls.
- [`../v5_semantic_localization/README.md`](../v5_semantic_localization/README.md)
  — the attempted fresh localization.
- [`../v5_inverse_evidence/README.md`](../v5_inverse_evidence/README.md) — the
  matched history/format test that became the strongest V5 result.

[Results map](../README.md) · [Project README](../../README.md)
