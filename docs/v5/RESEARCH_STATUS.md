# V5 research status

## Current result

RBG-4 is the first V5 result that passes its competence gate, effect-size threshold,
paired exact test, semantic control, and targeted replication cell. The central
finding is inverse evidence: redundant correct demonstrations selectively worsen
assertion-based strategic action while consequence reports remain stable. A table
representation removes the harm.

## Study sequence

| Study | Question | Verdict |
|---|---|---|
| RBG-1 | Can demonstrated receiver policies support revealed-belief games? | Failed competence gate; no claim |
| RBG-2 | Can explicit reportable consequences diverge from meaningful action? | Passed; 42.19-point semantic gap |
| RBG-3 | Does the large effect survive history-free table paraphrase and localize by surface? | Strong magnitude failed; seven assertion-specific, rehearsal-repaired errors |
| RBG-4 | Do redundant history and policy format explain the attenuation within matched games? | Passed; 23.96-point inverse-evidence effect, prose-specific |

## Open hypotheses

1. **Persistent policy-state hypothesis.** Redundant prose demonstrations create a
   target-matching assertion policy state that survives an explicit causal rule.
2. **Retrieval-gating hypothesis.** Consequence knowledge is available but not
   spontaneously recruited; same-trajectory rehearsal and tables restore access.
3. **Assertion-script hypothesis.** The model applies a learned “to cause false X,
   assert X” script even when a contrarian receiver reverses the result.
4. **Model-family hypothesis.** The effect may reflect Qwen post-training rather
   than transformer cognition generally; multi-model replication is essential.

## Licensed next experiment

Freeze a natural activation-interchange study using matched RBG-4 prompts. Patch
only naturally occurring states from correct table/opaque counterparts into failed
redundant-prose assertion prompts, with reverse and same-condition controls. Do not
use arbitrary steering vectors or tune layers against the final endpoint. A small
fresh behavioral replication should accompany any mechanistic result.

This experiment is now frozen as **V5-RBG-5**. It uses discovery data to choose
one site inside the pre-established layers 36–43, hashes that choice, and tests it
once on lexically disjoint locked games. Probes and optional J-space trajectories
remain secondary diagnostics.

## RBG-5 outcome

The fresh behavior run replicated the central action effect more strongly
(37.50 points, `p=.0000178`), retained 100% opaque actions, and again showed no
table harm. However, its absolute prose consequence-report gap was 5.208 points,
just above the frozen five-point ceiling. The conjunctive gate failed, so no
RBG-5 activation, probe, patch, or J-space output was opened. The internal cause
therefore remains unresolved.

## RBG-5B status

RBG-5B is now prospectively frozen as the powered mechanistic follow-up: 96 fresh
base games, exact BF16 residual capture, fixed-C probes, matched natural residual
interchange, and required post-patch J-space readouts. Its one-sided report
non-inferiority gate accepts improvements while still excluding report damage.
The dataset/config hashes and append-only artifact namespace are committed;
Modal execution is pending service connectivity.  A manual GitHub Actions
fallback is frozen at `.github/workflows/v5-mechanistic-decomposition-b.yml`;
it invokes the same Modal functions and downloads the mounted-volume artifacts
for offline analysis when the local Modal client cannot reach the control plane.
