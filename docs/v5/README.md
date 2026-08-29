# Protocol V5 — revealed-belief and inverse-evidence games

## What V5 was trying to do

V5 separated two abilities that are easy to confuse:

1. reporting what an action will cause; and
2. choosing the action that will achieve a goal.

The games used an explicit contrarian receiver: a claim of `RED` causes a
`BLUE` response, and vice versa. This lets us test action choice without
pretending that the model’s internal state is already a theory of mind.

## The sequence

| Study | Plain-language question | Verdict |
|---|---|---|
| RBG-1 | Can the model use the demonstrated receiver policy? | Competence gate failed. |
| RBG-2 | Can consequence reports stay correct while meaningful actions fail? | Passed; 42.19-point action gap. |
| RBG-3 | Does the large effect survive fresh surface controls? | Large replication failed; residual errors were informative. |
| RBG-4 | Do redundant history and prose format explain the effect? | Passed prospectively; inverse evidence. |
| RBG-5 | Can a matched natural activation repair the prose/assertion failure? | Prospectively frozen; not run. |

## Documents

- [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md) — current evidence and open hypotheses.
- [`revealed-belief-games-preregistration.md`](revealed-belief-games-preregistration.md)
  — RBG-1 design.
- [`semantic-override-preregistration.md`](semantic-override-preregistration.md)
  — RBG-2 design.
- [`semantic-localization-preregistration.md`](semantic-localization-preregistration.md)
  — RBG-3 design.
- [`inverse-evidence-preregistration.md`](inverse-evidence-preregistration.md)
  — RBG-4 design.
- [`mechanistic-decomposition-preregistration.md`](mechanistic-decomposition-preregistration.md)
  — RBG-5 discovery/locked natural-interchange design.
- [`decision-log.md`](decision-log.md) — append-only decisions.
- [`../../results/v5_inverse_evidence/README.md`](../../results/v5_inverse_evidence/README.md)
  — plain-language headline result.

[Documentation map](../README.md) · [Project README](../../README.md)
