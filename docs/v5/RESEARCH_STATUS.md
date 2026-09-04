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
| RBG-6 | Does seeing self-generated consequence reports control later action? | 0/48 primary dissociations; +58.33-point self-generated rescue; exploratory follow-up |

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

## RBG-5B outcome

RBG-5B was prospectively frozen as the powered mechanistic follow-up: 96 fresh
base games, exact BF16 residual capture, fixed-C probes, matched natural residual
interchange, and required post-patch J-space readouts. Its one-sided report
non-inferiority gate accepted improvements while still excluding report damage.
The locked salvage completed GPU capture and patching in 319.23 seconds, then ran
the geometry, probe, and 2,000-resample statistics locally on CPU. The natural
residual interchange repaired 2/50 held-out failures (4%), with exact clustered
sign-flip `p = 0.429238`; the preregistered 20%/positive-margin/`p<.05`
conjunction therefore failed. Consequence reports and non-damage controls stayed
at 100%, but the strict identity-margin control shifted by at most 0.5, so the
selective causal-transport claim is not supported for this checkpoint and
contrast.

The required J-space secondary run completed separately, producing 69,120 rows
(34,560 per split). Its projected action-token top-1 ranking was 50.96% on
discovery and 54.69% on locked rows (57.73% opposed, 42.56% aligned). This is a
weak observational readout, not a causal steering result; it cannot reopen patch
selection or rescue the negative locked endpoint. The first red workflow was a
missing frozen-probe-file error during local post-processing; immutable GPU
artifacts were recovered and analyzed without rerunning the model.

## RBG-6 outcome

Issue #11 requested the missing endogenous trajectory: the model reports both
consequences, those exact report tokens remain as assistant turns, and it then
chooses. The analysis plan froze the report-order factorial, oracle and swapped
transcript controls, and local tokenization before execution. On the RBG-2-like
strategic opposed assertion cell, both reports were correct in 46/48 trajectories
and none chose the wrong action. Self-generated action accuracy was 95.83% versus
37.50% for the immutable direct baseline (58.33-point gain, exact clustered
`p=.000061`). Oracle replay was 100%; swapped replay followed the inverted
report-implied action in 43/48 cases. The matched opaque control stayed at 100%.

This supports behavioral trajectory dependence in this prompt family, not a claim
about deception or a particular internal mechanism. See
[`../../results/v5_full_action_trajectory/FINDINGS.md`](../../results/v5_full_action_trajectory/FINDINGS.md).
