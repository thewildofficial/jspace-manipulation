# V5-RBG-4 findings: redundant correct evidence harms strategic action

## Verdict

All prospectively frozen RBG-4 gates passed. In explicit contrarian games with an
opposed goal, adding four correct demonstrations reduced meaningful-assertion
action accuracy from 79.17% to 55.21% while leaving option-consequence report
accuracy unchanged. The 23.96-point history harm was significant across 24 fresh
base games (two-sided exact cluster sign-flip p=.000244).

The effect was semantic rather than a generic cost of extra context. Opaque-token
actions were 100% correct, and the assertion-versus-opaque difference-in-differences
was 23.96 points (p=.000244). A compact consequence table eliminated the history
harm, whereas a prose rule amplified it sharply.

## Frozen outcomes

| Outcome | Result | Frozen gate |
|---|---:|---:|
| Opaque action accuracy | 384/384 (100%) | at least 95% |
| All consequence reports | 1520/1536 (98.96%) | at least 95% |
| Opposed assertion, no history | 76/96 (79.17%) | contrast arm |
| Opposed assertion, redundant history | 53/96 (55.21%) | contrast arm |
| No-history minus history gap | 23.96 points | at least 20 points |
| Consequence-report gap | 0.00 points | absolute gap at most 5 points |
| Exact clustered p | .000244 | below .05 |
| Assertion-minus-opaque history harm | 23.96 points | at least 20 points |
| Exact RBG-2-like action cell | 9/24 (37.50%) | at most 50% |
| Matched opaque action cell | 24/24 (100%) | at least 90% |
| RBG-2-like consequence reports | 44/48 (91.67%) | at least 90% |

## Representation interaction

The history effect depended on how the identical current policy was represented:

| Opposed meaningful assertions | No history | Four correct demonstrations | Change |
|---|---:|---:|---:|
| Prose mapping | 36/48 (75.00%) | 13/48 (27.08%) | −47.92 points |
| Consequence table | 40/48 (83.33%) | 40/48 (83.33%) | 0.00 points |

Thus more correct evidence is not intrinsically harmful. Redundant narrative
examples plus a prose causal rule produce the failure; tabularizing the same rule
abolishes it.

## Strong dissociation subset

Fifty-two opposed-assertion trials had both option consequences reported correctly
but a wrong action. All 52 selected the assertion whose content equaled the desired
receiver/device response even though the explicit contrarian rule said that this
action caused the other response. This is a behavioral knowledge/action
dissociation. It does not by itself prove that the reported consequence was encoded
or attended to at the exact moment of direct action selection.

## Claim boundary

The supported result is:

> In Qwen3.6-27B forced-choice contrarian games, redundant correct demonstrations
> can selectively impair assertion-based strategic action without impairing
> independent consequence reports; a consequence-table representation removes the
> impairment.

The study does not establish conscious deception, scheming, a general property of
other models, or an internal mechanism. Multi-model replication and a frozen
natural activation-interchange experiment remain required.

## Cost and provenance

- Dataset SHA-256:
  `3fdc9977f03700dbfb2118414ec9602a4e76d9d6e2b35788a81bc6acf4a1f5d1`
- Git commit: `1cb93e279a3f0ef0ddc8f0a67a3b0bbf43a4d904`
- GitHub Actions run: `32666720469`
- Modal run: `7c06999f4ad54d06b4454584e04faf9d`
- A100 elapsed time: 319.797 seconds
- Measured cost: USD 0.27817
- Buffered cost: USD 0.33381
- Raw payload SHA-256:
  `4e7ba1d0dc16e1fe231d7fe312e77dbede14e33e5db2eca2c6e030ed9237b4f9`
