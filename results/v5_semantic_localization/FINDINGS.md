# V5-RBG-3 findings: strong replication failed, residual localized

## Verdict

The fresh localization study did not reproduce the preregistered magnitude of
RBG-2. Opposed assertion accuracy was 85.42%, above the frozen maximum of 50%, and
the aligned-minus-opposed gap was 14.58 points rather than the required 40. The
confirmatory replication and causal-rehearsal hypotheses are therefore not
promoted.

The residual errors were nevertheless selective. Opaque tokens, non-assertive
quoted strings, and labeled buttons were each 100% correct on opposed goals. All
seven assertion errors occurred in the strategic frame, had both consequences
reported correctly, selected the claim whose content matched the target but whose
stated consequence did not, and were repaired by same-trajectory causal rehearsal.
These are sub-threshold diagnostics that motivate a direct history/format ablation.

## Frozen outcomes

| Outcome | Result | Frozen gate |
|---|---:|---:|
| Direct opaque actions | 96/96 (100%) | at least 95% |
| Opaque consequence reports | 192/192 (100%) | at least 95% |
| Assertion consequence reports | 189/192 (98.44%) | at least 90% |
| Aligned assertion actions | 48/48 (100%) | at least 90% |
| Opposed assertion actions | 41/48 (85.42%) | at most 50% |
| Aligned-minus-opposed gap | 14.58 points | at least 40 points |
| Exact clustered p | .015625 | below .05 |
| Rehearsed opposed assertions | 48/48 (100%) | at least 75% |
| Rehearsal gain | 14.58 points | at least 25 points |

The assertion-specific and rehearsal contrasts were statistically nonzero but did
not meet their prospectively frozen effect-size thresholds. They must not be
reported as confirmatory successes.

## Interpretation

RBG-2's large effect is not invariant to fresh history-free table wording. RBG-3
rules out a broad lexical-target account in its own prompt family: actions that
contained the target concept without making an assertion were 100% correct. It
leaves two plausible sources for the cross-study attenuation:

1. the four redundant receiver-policy demonstrations in RBG-2; and
2. prose mapping versus a compact consequence table.

A fresh within-study factorial must manipulate these variables rather than compare
RBG-2 and RBG-3 post hoc.

## Cost and provenance

- Dataset SHA-256:
  `c38769c84a1129e1ace7b44bc055be085fc6e087d360cd7efffbf8d85891ce3d`
- Git commit: `1071c9ef1e235c3dada61f5b97575958fb247a66`
- GitHub Actions run: `32666221006`
- Modal run: `b5f9fae55ecf452aa9495be25793750c`
- A100 elapsed time: 207.117 seconds
- Measured cost: USD 0.18016
- Buffered cost: USD 0.21619
- Raw payload SHA-256:
  `296b430a860a4cf6b702b365e359e5c54c3ced4ecd1b8a1133bf1b72debd2ca2`

