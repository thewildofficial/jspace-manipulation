# Causal manipulation of reporting policies in J-space

Can a language model continue to represent what is true while a causal
intervention changes the policy it uses to report that truth?

This repository tests that narrow mechanistic question in
`Qwen/Qwen3.6-27B` with the pinned Anthropic Jacobian Lens. It does not equate
instructed concealment with real-world deception, model intent, or
consciousness.

> **Current headline: V5 found a prospectively confirmed inverse-evidence effect.**
> In explicit contrarian games, adding four correct redundant demonstrations
> selectively made meaningful strategic actions worse without making consequence
> reports worse. The harm was specific to prose/assertion representations and
> disappeared with an equivalent consequence table or opaque actions. This is a
> behavioral knowledge/action dissociation in one model, not yet an internal
> mechanism or evidence of scheming.

## Headline result: more correct evidence can impair strategic action

V5 asks whether a model can correctly state what each action will cause yet fail
to use that causal knowledge when choosing strategically. The cleanest example uses
a contrarian receiver:

- claiming `RED` makes the receiver choose `BLUE`;
- claiming `BLUE` makes the receiver choose `RED`; and
- the sender is rewarded if the receiver chooses a specified target.

Qwen3.6-27B could report these option consequences at 98.96% accuracy. However,
for opposed goals with meaningful assertions, adding four perfectly correct and
redundant demonstrations reduced action accuracy from 79.17% to 55.21%. The
23.96-point paired effect passed its frozen threshold and exact base-game test
(`p=.000244`), while the consequence-report gap was exactly zero.

The representation interaction was larger:

| Explicit contrarian opposed assertions | No history | Four correct demonstrations |
|---|---:|---:|
| Prose causal rule | 75.00% | 27.08% |
| Consequence table | 83.33% | 83.33% |

Matched opaque-token actions remained 100% correct. Fifty-two trials had both
option consequences reported correctly but a wrong action; all 52 selected the
assertion whose words matched the desired response even though the stated
contrarian rule implied the opposite outcome.

The supported claim is narrow: **in this Qwen checkpoint and forced-choice game
family, redundant correct narrative evidence can worsen assertion-based strategic
action without degrading independent causal reports, and tabularizing the same rule
removes the harm.** It does not establish consciousness, deliberate deception,
scheming, or cross-model generality.

- [RBG-4 frozen findings](results/v5_inverse_evidence/FINDINGS.md)
- [V5 research status and open hypotheses](docs/v5/RESEARCH_STATUS.md)
- [Preregistration](docs/v5/inverse-evidence-preregistration.md)
- [Literature positioning](docs/v5/literature-positioning.md)
- [Append-only V5 decision log](docs/v5/decision-log.md)

## What happened

The early V1 work established useful but observational prerequisites:

- the initial 4B candidate failed its balanced behavioral screen and was
  retired;
- Qwen3.6-27B passed the toy behavioral gate (192/192 rows);
- the released 27B lens passed structural/reference checks;
- a balanced discovery pilot found a middle-layer policy-family signal under
  named and unlabeled rules.
- a separate 27B causal-discovery study found the candidate signal readable but
  not writable: no practically meaningful, fact-preserving reporting-policy
  mechanism was established.

V2 then required a causal instrumentation gate before any strategic task:

1. **Original H0 failed.** Numerical transport, direction, and analytic-swap
   checks passed, but the preregistered workspace-band/all-position topology
   produced 0/90 target-answer top-1 outcomes. An exploratory one-layer,
   all-position topology produced 25/90 overall and 24/33 on countries. The
   original verdict remains: **no strategic-reporting experiment is licensed
   or was run**.
2. **H0R diagnosed the topology.** Three burned-country GPU diagnostics found
   exact fp32 coordinate exchange and a dose-responsive lower-strength regime.
   The frozen selection rule chose layers 36--42, argument-through-end
   positions, and alpha 0.5. On burned data it produced mean target-vs-source
   log-odds gain +4.34 with low KL and displacement.
3. **The first fresh H0R-C corpus was invalid.** Baseline accuracy was 60.8%,
   below the frozen 80% gate, so its runner stopped before intervention.
4. **Replacement H0R-C failed prospectively.** The simplified fresh corpus
   passed baseline at 125/130 (96.2%). The frozen semantic intervention had
   mean log-odds gain +3.53 (95% scenario-bootstrap interval +2.68 to +4.29),
   91.2% positive direction, and a +2.82 advantage over random/unrelated
   controls, while remaining inside all validity limits. However, only 4/125
   eligible trials made the target answer top-1 (3.2%), below the
   preregistered 20% minimum. Because the gate was conjunctive, H0R-C failed.
5. **Stage 1 found residual state information but failed its J-space endpoint.**
   Locked behavior passed at 100% for explicit state and 98.75% for inferred
   state. Locked residual state probes reached 100% and 99.2%, respectively.
   However, transformed-report J-space K was -0.108 (95% CI -0.142 to -0.073)
   in Stage 1A and -0.017 (95% CI -0.043 to 0.008) in Stage 1B. Neither met the
   frozen positive-retention criterion.
6. **V2-E1 found a generic decision abstraction, not strategic intent.** Across
   locked Kuhn and signaling renderings, `optimal`/`optimize` J-lens tokens
   appeared in every row at layer 43 and preceded late action commitment. The
   five-game residual-over-J-space/output strategy contrast failed, and
   same-action Kuhn bluff/thin/value decoding was at chance.

The strong directional effect is a useful mechanistic diagnostic, but it does
not satisfy the locked criterion for reliable counterfactual control. H0R-D
was prohibited by design. Continuing the causal study now requires a new
preregistration, model/lens change, or redesigned instrument—not further tuning
on the strategic task.

## Evidence map

| Stage | Result | Evidence |
|---|---|---|
| 4B behavioral screen | Failed and retired | [`results/4b_behavioral_screen/`](results/4b_behavioral_screen/README.md) |
| 27B behavioral gate | Passed, 192/192 | [`results/27b_behavioral_gate/`](results/27b_behavioral_gate/README.md) |
| 27B lens integrity | Passed structural/reference checks | [`results/27b_lens_integrity/`](results/27b_lens_integrity/README.md) |
| Observational pilot | Candidate signal; no causal claim | [`results/27b_observational_pilot/`](results/27b_observational_pilot/README.md) |
| Causal discovery | Readable, not writable; clean negative | [`results/27b_causal_discovery/`](results/27b_causal_discovery/README.md) |
| V2 H0 | Failed causal topology | [`results/v2_smoke_tests/`](results/v2_smoke_tests/README.md) |
| H0R-B diagnostic | Candidate selected on burned controls | [`results/v2_h0r_diagnostic/`](results/v2_h0r_diagnostic/) |
| First H0R-C corpus | Invalid 60.8% baseline; intervention unopened | [`results/v2_h0r_argument_validation/`](results/v2_h0r_argument_validation/README.md) |
| Replacement H0R-C | Failed 20% target-top-1 criterion | [`results/v2_h0r_argument_validation_v2/`](results/v2_h0r_argument_validation_v2/README.md) |
| H0R-D / strategic V2 | Not run | [`docs/v2/h0r-final-report.md`](docs/v2/h0r-final-report.md) |
| Stage 1 state-report dissociation | J-space criterion failed; residual probes positive | [`docs/v2/stage1-final-report.md`](docs/v2/stage1-final-report.md) |
| V2-E1 Strategic Workspace Atlas | Generic optimization signal replicated; specific strategy endpoints failed | [`docs/v2/strategic-workspace-atlas-final-report.md`](docs/v2/strategic-workspace-atlas-final-report.md) |
| V4 strategic epistemic search | Correct-rationale report penalty replicated; binding mechanism not established | [`docs/v4/research-status.md`](docs/v4/research-status.md) |
| V4 ordinal permutation | Behavior 345/360; broad truth suppression; strategic-frame lead is exploratory | [`results/v4_strategic_epistemic_search/FINDINGS.md`](results/v4_strategic_epistemic_search/FINDINGS.md) |
| V5 RBG-1 | Receiver-policy competence gate failed; no strategic claim | [`results/v5_revealed_belief_games/FINDINGS.md`](results/v5_revealed_belief_games/FINDINGS.md) |
| V5 RBG-2 | Explicit consequence reports diverged from meaningful action; 42.19-point semantic gap | [`results/v5_semantic_override/FINDINGS.md`](results/v5_semantic_override/FINDINGS.md) |
| V5 RBG-3 | Strong replication magnitude failed; seven assertion-specific errors were rehearsal-repaired | [`results/v5_semantic_localization/FINDINGS.md`](results/v5_semantic_localization/FINDINGS.md) |
| V5 RBG-4 | Inverse-evidence and semantic-specificity gates passed prospectively | [`results/v5_inverse_evidence/FINDINGS.md`](results/v5_inverse_evidence/FINDINGS.md) |

The append-only methodological record is in
[`docs/v2/decision-log.md`](docs/v2/decision-log.md). The original H0 and H0R
preregistrations, analysis rules, diagnostic interpretation, and terminal
report are under [`docs/v2/`](docs/v2/).

V4 has a separate append-only record in
[`docs/v4/decision-log.md`](docs/v4/decision-log.md), a ruthless current-state
audit in [`docs/v4/research-status.md`](docs/v4/research-status.md), and a ranked
literature-facing agenda in
[`docs/v4/open-hypotheses.md`](docs/v4/open-hypotheses.md).

V5 has an append-only record in
[`docs/v5/decision-log.md`](docs/v5/decision-log.md), a reader-facing synthesis in
[`docs/v5/RESEARCH_STATUS.md`](docs/v5/RESEARCH_STATUS.md), and explicit novelty
boundaries in
[`docs/v5/literature-positioning.md`](docs/v5/literature-positioning.md).

## Claim boundary

A readable J-space feature without a selective causal effect is observational,
not a mechanism. A directional log-odds change without reliable target-answer
conversion is evidence of partial causal access, not a validated instrument.
The project required both unseen argument substitution (H0R-C) and a computed
intermediate (H0R-D) to pass prospectively before testing latent truth and
strategic reporting. That conjunction was not met.

The V1--V2 mechanistic work supports six narrow conclusions:

- the pinned J-lens transport and coordinate-write implementation are
  numerically sound;
- Qwen has a measurable, selective, strength-dependent response to the frozen
  J-space write;
- that write did not achieve the preregistered level of reliable
  counterfactual control needed for the strategic study.
- task-state information is prospectively decodable from the residual stream
  on Stage 1's controlled held-out families, but the pinned J-space readout did
  not retain positive true-state evidence under transformed reporting.
- in V2-E1, a prompt-absent generic optimization token family became J-space
  visible before stable output-action preparation in locked Kuhn and signaling
  renderings;
- V2-E1 did not show reliable, rendering-general decoding of the specific
  strategy behind a shared action.

Separately, V5 supports a behavioral conclusion: redundant correct narrative
evidence can selectively impair meaningful assertion-based action while independent
consequence reports remain stable, and an equivalent table representation can
remove that impairment. V5 has not yet established the internal mechanism.

## Repository map

```text
src/jspace_policy/
  dataset.py              deterministic factorial prompt generation
  analysis.py             paired estimates and bootstrap intervals
  interventions.py        J-lens directions and residual hooks
  h0r.py                   locked positive-control generators
  h0r_diagnostics.py       coordinate, conditioning, and reconstruction checks
  stage1.py                frozen four-state Stage 1 corpus
  stage1_analysis.py       bootstrap and probe metrics
  workspace_atlas.py       exact game solvers and atlas corpus
scripts/
  analyze_results.py       V1 summaries and figures
  analyze_v2_h0.py         original H0 summaries and figures
  analyze_h0r.py           H0R-B selection tables and nine diagnostic figures
  analyze_stage1.py        Stage 1 tables, figure, and generated report
  analyze_workspace_atlas.py V2-E1 atlas and locked endpoints
modal_app.py               V1 benchmark/readout entry points
modal_v2.py                original bounded H0 GPU runner
modal_h0r.py               H0R freeze, diagnostic, and prospective runners
modal_stage1.py            separated Stage 1 behavior/readout entry points
modal_workspace_atlas.py   V2-E1 behavior and all-layer readout runner
configs/v2/                preregistered gates and immutable H0R controls
docs/v2/                   V2/H0/H0R plans, decisions, and reports
results/                   committed raw evidence, summaries, and figures
```

## Local verification and analysis

Set up the project and run the non-GPU checks:

```bash
uv sync --extra dev --extra modal
uv run pytest
```

Regenerate committed analyses from their immutable raw rows:

```bash
uv run python scripts/analyze_v2_h0.py
uv run python scripts/analyze_h0r.py
uv run python scripts/analyze_stage1.py --phase locked
uv run python scripts/analyze_workspace_atlas.py --phase open
uv run python scripts/analyze_workspace_atlas.py --phase locked
```

The H0R analysis emits the full layer curve, top-1 curve, position and strength
sweeps, cumulative-layer comparison, coordinate trajectory, reconstruction
curve, effect-versus-distance plot, and semantic-versus-control comparison in
[`results/v2_h0r_diagnostic/figures/`](results/v2_h0r_diagnostic/figures/).

The Modal entry points preserve the exact GPU workflow used for the committed
runs. Prospective entry points intentionally refuse to overwrite existing
result directories. Do not rerun or retune them as though the existing locked
controls were fresh.

## Interpretation and novelty

Activation probes and steering directions for truthfulness or deception already
exist, and the Jacobian Lens work already demonstrates causal concept swaps.
The proposed novelty here was the stronger combined test of a reporting policy:
a vocabulary-grounded J-space intervention, factorial world-state controls, a
required compositional output change, and independent fact preservation.

That full claim was not reached. The retained result is a transparent negative
and diagnostic study: readable and directionally causal structure did not
become a sufficiently reliable instrument for the planned latent-knowledge and
strategic-reporting experiment.

## Primary references

- Gurnee et al., [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/) and the [reference implementation](https://github.com/anthropics/jacobian-lens).
- Zou et al., [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405).
- Burns et al., [Discovering Latent Knowledge in Language Models Without Supervision](https://arxiv.org/abs/2212.03827).
- Levinstein and Herrmann, [Still No Lie Detector for Language Models](https://arxiv.org/abs/2307.00175).
- Wang, Zhang, and Sun, [When Thinking LLMs Lie](https://arxiv.org/abs/2506.04909).
