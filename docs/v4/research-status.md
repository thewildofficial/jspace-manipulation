# V4 research status

**Status date:** 2026-08-24

**Branch:** `research/v4-strategic-epistemic-search`

**Pinned model:** `Qwen/Qwen3.6-27B` at
`6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`

## Executive verdict

V4 has one replicated behavioral result: retaining a correct rationale reduces
later forced-choice access to a simple relation that answer-only context preserves.
The fresh ordinal study confirms a broad rationale-induced shift away from the true
response. It does not identify the originally suspected label or positional
binding mechanism.

This is scientifically real but not yet a high-impact mechanistic or alignment
result. The current task prints the receiver policy, so it is not genuine theory of
mind. No J-space readout or intervention has been run in V4. No result supports
deception, scheming, introspection, or hidden intent.

The highest-upside lead is a post-hoc but large strategic-frame interaction on
isomorphic numeric games. That lead needs one cheap fresh confirmation before any
mechanistic spend.

## Evidence ledger

| Question | Current answer | Evidence strength | Next decision |
|---|---|---|---|
| Can the model solve the games? | Yes: 234/240 original; 345/360 fresh ordinal | Prospective, strong | Competence is not the bottleneck |
| Does retaining the rationale hurt predicted-response report? | Yes | Discovery plus held-out plus fresh prospective study | Treat as established in this model |
| Is the effect uniquely self-generated? | No evidence | Matched unrelated trajectories were at least as harmful | Do not use “self-generated” as a mechanism claim |
| Is C→R3 lexical binding the cause? | Not established | Old alias test was upstream-incomplete; fresh label specificity Holm p=.191 | Downgrade |
| Is display-position binding the cause? | Not established | Fresh position specificity p=.290 | Downgrade |
| Is there generic truth-response suppression? | Yes | +1.1097 logits; 20/20 clusters; post-hoc characterization of a prospective effect | Replicate with a specificity-aware endpoint |
| Is strategic framing worse than device framing? | Promising but exploratory | +0.697 strategic interaction; 19/20 clusters; post-hoc p=3.81e-6 | Highest-priority fresh replication |
| Does truth remain outside J-space? | Unknown | No V4 internal-state data | Test only after frame replication |
| Can a causal patch repair the report? | Unknown | No V4 intervention | Narrow patch only after localization |
| Is this theory of mind? | No | Receiver policy is explicitly printed | Build inferred-receiver task later |
| Is this deception or scheming? | No | No false message chosen from opposed incentives and inferred beliefs | Requires a new game |

## What changed after the ordinal result

The frozen primary endpoint compared each designated lure with the true response.
Both label and position lures moved strongly. The frozen interpretation table maps
this to a mixed binding code, but the endpoint does not distinguish selective lure
capture from a general fall in the truth logit. A post-hoc comparison against the
nondesignated false response found no corrected evidence for either binding type.

This is a design lesson, not an excuse to discard the frozen result. The repository
reports both:

1. the preregistered result exactly as specified; and
2. the later diagnostic that limits its mechanistic interpretation.

## Protocol integrity

The ordinal behavior artifact unintentionally contained the preregistered report
rows because the shared behavior remote also executed `_self_report_rows`. The
behavior gate passed, all report prompts and analyses were already frozen, and the
artifact was immutable before inspection. The report is retained as prospective,
with the stage-separation deviation disclosed.

The code is fixed so future behavior entrypoints explicitly disable report
generation. Raw evidence is never overwritten.

## Cost state

Nine V4 GPU runs have cost $2.08010 measured and $2.49612 under buffered
accounting. The V4 protocol ceiling is $18.50 buffered; the user's overall allowance
was approximately $22. No mechanistic sweep has consumed budget.

## Canonical artifacts

- Original V4 plan: `docs/v4/strategic-epistemic-search-plan.md`
- Append-only decisions: `docs/v4/decision-log.md`
- Open hypotheses: `docs/v4/open-hypotheses.md`
- Living findings: `results/v4_strategic_epistemic_search/FINDINGS.md`
- Frozen ordinal protocol:
  `configs/v4/strategic_epistemic_search/ordinal_binding_permutation.json`
- Frozen ordinal dataset:
  `configs/v4/strategic_epistemic_search/ordinal_binding_dataset.json`
- Immutable source payload:
  `results/v4_strategic_epistemic_search/raw/ordinal_binding_behavior_v1.json`
- Lossless embedded-report extraction:
  `results/v4_strategic_epistemic_search/raw/ordinal_binding_report_v1_from_behavior.json`
- Frozen primary analysis:
  `results/v4_strategic_epistemic_search/analysis/ordinal_binding_report_v1_preregistered_analysis.json`
- Explicitly post-hoc diagnostic:
  `results/v4_strategic_epistemic_search/analysis/ordinal_binding_report_v1_posthoc_diagnostics.json`

## Stop rules

- Do not call the current task deception, scheming, or theory of mind.
- Do not infer a J-space mechanism from report behavior.
- Do not claim mixed binding from the two lure-vs-true p-values.
- Do not tune a mechanism on these 20 games and then present it as confirmatory.
- Do not run a broad layer × position × direction sweep.
- If the strategic-frame interaction fails a fresh replication, archive V4 as a
  modest rationale-interference result and pivot to an inferred-receiver game.
- If it replicates, freeze a small causal localization and repair experiment before
  viewing any V4 internal states.
