# V5-RBG-2 findings: semantic action-outcome override passed

## Verdict

Qwen3.6-27B passed every prospectively frozen gate for semantic action-outcome
override. With an explicit contrarian response map, opaque-token actions were
100% correct, while semantically meaningful claim actions were 57.81% correct
despite 97.66% accuracy when independently reporting those claims' consequences.
The 42.19 percentage-point gap was consistent across all 16 fresh base games
(two-sided exact cluster sign-flip p=.0000305).

This is a behavioral knowledge/action dissociation, not yet a mechanistic result.
It shows that explicit, reportable causal knowledge can fail to control action
selection when the action is a meaningful assertion and the goal is to produce a
false receiver conclusion.

## Frozen outcomes

| Outcome | Result | Frozen gate |
|---|---:|---:|
| Explicit opaque action accuracy | 64/64 (100%) | at least 90% |
| Minimum opaque control cell | 8/8 (100%) | at least 80% |
| Explicit opaque consequence reports | 128/128 (100%) | at least 90% |
| Explicit contrarian opaque actions | 64/64 (100%) | at least 85% |
| Explicit contrarian claim actions | 37/64 (57.81%) | at most 65% |
| Explicit contrarian claim reports | 125/128 (97.66%) | at least 85% |
| Opaque-minus-claim action gap | 42.19 percentage points | at least 20 points |
| Exact base-game sign-flip p | .0000305 | below .05 |

Twenty-four claim trials had both option consequences reported correctly but a
wrong action. In all 24, the selected claim's content equaled the desired receiver
response even though the prompt explicitly said that the contrarian receiver would
choose the other state.

## Selective structure

The effect is not generic difficulty with contrarian mappings. It is isolated to
goals that oppose the private true state:

| Explicit contrarian claims | Aligned goal | Opposed goal |
|---|---:|---:|
| Strategic receiver | 16/16 (100%) | 0/16 (0%) |
| Non-agentic device | 16/16 (100%) | 5/16 (31.25%) |

The corresponding explicit opaque-token cells were all 16/16. Thus the model can
use the same contrarian map perfectly when the desired response is true, and can
use it perfectly for arbitrary tokens regardless of the desired response. When a
meaningful assertion is used to induce a false response, action selection instead
resembles a surface strategy of asserting the false target. This interpretation is
diagnostic rather than confirmatory because the aligned/opposed decomposition was
not separately thresholded in the frozen analysis.

## Scope and next tests

RBG-2 does not show that the model consciously knows and ignores the consequence,
nor does an independent forced-choice report guarantee that identical knowledge
was active during action selection. Fresh replication must separate three candidate
causes:

1. lexical overlap between an action label and the target;
2. assertion/deception semantics beyond lexical overlap; and
3. failure to retrieve the causal map during choice, testable by causal rehearsal.

Only after those behavioral controls should natural activation interchange be used
to test whether a consequence representation can be transplanted from a matched
opaque-token prompt to repair meaningful-claim action selection.

## Cost and provenance

- Preregistered dataset SHA-256:
  `4e5b1c12ae40889fdcbb23c2226e3deaf5ac6027644f481cce92b3866ab2d9f6`
- Git commit: `758c5a0e8876c8c0a1103700618e1d572c52f26c`
- GitHub Actions run: `32665546622`
- Modal run: `e0f8fa12f60841b89b79669cbedc9adb`
- A100 elapsed time: 241.837 seconds
- Measured cost: USD 0.21036
- Buffered cost: USD 0.25243
- Raw payload SHA-256:
  `d46a60a7446d485af64277b783d44942a725a59416eb5e9d39f7902e23956c25`

