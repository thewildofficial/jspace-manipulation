# J-Space Manipulation

This repository asks a simple question about language models:

> Can a model keep one fact in mind while changing the answer it gives because a reporting rule changed?

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
- [RBG-5 frozen mechanistic protocol](docs/v5/mechanistic-decomposition-preregistration.md)
- [V5 research status and open hypotheses](docs/v5/RESEARCH_STATUS.md)
- [Preregistration](docs/v5/inverse-evidence-preregistration.md)
- [Literature positioning](docs/v5/literature-positioning.md)
- [Append-only V5 decision log](docs/v5/decision-log.md)

Here is the same question in its smallest form. The hidden fact is **A** or
**B**. Under a **COPY** rule the model should say the fact; under a **FLIP** rule
it should say the other answer. We want to know whether the model keeps the
fact and the reporting rule separate, and whether we can change one without
destroying the other.

The project looks for those two pieces inside the model using a tool called a
**Jacobian Lens**. The lens gives us a vocabulary for looking at intermediate
states in the model. We then make small, carefully controlled changes in that
space and check what changes in the answer.

This is a research record. It does not claim that the model is conscious,
deceptive, or “lying” in a human sense.

## The short version

The project contains both positive and negative results, and the order matters:

| Question | What happened |
|---|---|
| Can the smaller 4B model follow the toy reporting task? | No. One balanced cell failed, so we retired it. |
| Can the 27B model follow the same task? | Yes: 192/192 rows. |
| Does the released lens look structurally sound? | Yes, on structural and reference checks. |
| Is there an observable policy-related signal? | Yes, in an exploratory pilot; this was not causal evidence. |
| Can the original V2 intervention reliably swap an answer? | No: 0/90 target top-1 outcomes for the required topology. |
| Did the V2 recovery fix that? | No: 4/125 target top-1 outcomes, below the locked 20% gate. |
| Was the original strategic J-space experiment run? | No. The gates stopped it. |
| Did later behavioral game studies find anything? | Yes. V5-RBG-4 found the inverse-evidence effect described above. |
| Has the internal cause been tested? | No. V5-RBG-5 replicated behavior, but missed its report-stability gate by 0.208 points, so activations stayed closed. |

The V2 causal-instrument branch is closed. The later V3–V5 protocols are
separate follow-on studies and must not be used to rewrite the V2 verdict.

## How the names work

The repository uses different labels for different kinds of change. They are
not interchangeable:

| Label | Meaning | Example |
|---|---|---|
| **Research Stage** | The scientific difficulty ladder. | Stage 1: toy mechanism |
| **Protocol generation** | A chronological experimental program. | V2: causal instrument |
| **Study** | One concrete question inside a protocol. | V5-RBG-4: inverse evidence |
| **Phase** | A step inside one study. | V3-JI1 Phase C: fresh corpus freeze |
| **Gate** | A pass/fail checkpoint. | V2-H0 |
| **Hypothesis / recovery** | A locked claim or a recovery attempt. | H0R-B, H0R-C |
| **Prompt revision** | A wording change inside one experiment. | P1–P6; historical files use `v1`–`v6` |

The three scientific stages are:

1. **Stage 1 — Toy mechanism:** an explicit fact plus a simple reporting rule. This is the stage documented here; it currently ends with a failed causal-instrument gate.
2. **Stage 2 — Controlled composition:** the model must infer the fact from a short story and then apply the reporting rule. Not reached.
3. **Stage 3 — Naturalistic behavior:** the model must infer both the relevant fact and an instrumentally useful reporting policy. Not reached.

The 4B and 27B behavioral experiments are therefore both part of Stage 1. The
27B experiment is **Stage 1B**, a scale-up after the 4B attempt, not a separate
scientific stage. The names `V1` through `V5` refer to protocol generations,
not additional research stages. V4 and V5 are later behavioral follow-ons; they
do not retroactively open Stage 2 or Stage 3 of the original mechanistic ladder.

| Protocol | What it did |
|---|---|
| **V1 — foundation** | Checked task competence, lens integrity, and observable policy signals. |
| **V2 — causal instrument** | Tested whether a J-space edit could reliably change a fresh answer; added controlled workspace follow-ons. |
| **V3 — intervention replication** | Tried to characterize the edit on a new clean corpus; stopped before fresh intervention. |
| **V4 — rationale interference** | Tested whether keeping a correct explanation harms a later relational report. |
| **V5 — revealed-belief / inverse evidence** | Tested whether consequence knowledge can stay accurate while strategic action fails. |

## Read the project as a story

If you are new to the project, use this order:

1. [`results/README.md`](results/README.md) is the visual table of contents for every experiment and result.
2. [`docs/README.md`](docs/README.md) explains the research question, rules, plans, and decisions.
3. Pick a stage from the table below and read its folder README.
4. Open the linked raw data, summary tables, figures, code, or technical report when you want to check the details.

## Research path

The work was deliberately staged. Each stage had to earn the right to open the
next one.

| Canonical name | Plain-language question | Result | Start here |
|---|---|---|---|
| **Stage 1A — 4B behavioral screen** | Can a small model apply a tiny “copy or flip” rule? | Failed one cell; model retired. | [`results/4b_behavioral_screen/`](results/4b_behavioral_screen/README.md) |
| **Stage 1B — 27B scaled behavioral gate** | Does the larger model reliably do the same toy task? | Passed all 192 rows. | [`results/27b_behavioral_gate/`](results/27b_behavioral_gate/README.md) |
| **Stage 1C — released-lens integrity** | Does the lens fit the selected model and produce sensible readouts? | Passed structural and reference checks. | [`results/27b_lens_integrity/`](results/27b_lens_integrity/README.md) |
| **Stage 1D — observational policy-signal pilot** | Do internal readouts contain a possible reporting-policy signal? | A candidate signal appeared; exploratory only. | [`results/27b_observational_pilot/`](results/27b_observational_pilot/README.md) |
| **Stage 1E.1 — V2 workspace mapping** | Which middle layers did the task-independent rule select? | Layers 36–43 were selected before flexible results were opened. | [`results/v2_workspace_mapping/`](results/v2_workspace_mapping/README.md) |
| **Stage 1E.2 — V2 causal gate (H0)** | Does the planned intervention work on held-out country-answer controls? | Failed the required persistent, full-band topology. | [`results/v2_smoke_tests/`](results/v2_smoke_tests/README.md) |
| **Stage 1E.3 — V2 recovery diagnosis (H0R-B)** | Why might the first intervention topology have failed? | A lower-strength, local candidate was frozen for a fresh test. | [`results/v2_h0r_diagnostic/`](results/v2_h0r_diagnostic/README.md) |
| **Stage 1E.4a/b — V2 prospective validation (H0R-C)** | Does that frozen candidate reliably move an unseen answer? | The first corpus was invalid; the replacement reached 4/125 target top-1. | [`results/v2_h0r_argument_validation_v2/`](results/v2_h0r_argument_validation_v2/README.md) |
| **V2-SR-1 — state/report dissociation** | Is state information still readable when the reporting rule changes? | Residual probes were positive; the J-space endpoint failed. | [`results/v2_stage1/`](results/v2_stage1/README.md) |
| **V2-SWA-1 — strategic workspace atlas** | Does a generic optimization signal identify the specific strategy? | Generic signal replicated; strategy decoding did not. | [`results/v2_workspace_atlas/`](results/v2_workspace_atlas/README.md) |
| **V2-ST-1 — strategic J-Lens trajectories** | Can top-k lens trajectories expose the decisive pathway? | Generic task semantics and late answer preparation only. | [`results/v2_strategic_trajectories/`](results/v2_strategic_trajectories/README.md) |
| **V3-JI1 — J-space intervention replication** | Can a fresh corpus support a cleaner intervention test? | Burned engineering gate passed; fresh corpus freeze failed before intervention. | [`results/v3_jspace_interventions/`](results/v3_jspace_interventions/README.md) |
| **V4-SES-1 — strategic epistemic search** | Does retaining a correct rationale hurt a later relational report? | Yes, replicated behaviorally; the internal mechanism is unknown. | [`results/v4_strategic_epistemic_search/`](results/v4_strategic_epistemic_search/README.md) |
| **V5-RBG-1** | Can the model use a demonstrated receiver policy in a game? | No; competence gate failed. | [`results/v5_revealed_belief_games/`](results/v5_revealed_belief_games/README.md) |
| **V5-RBG-2/3/4** | Does the action/report gap survive controls and matched history? | RBG-2 found the gap, RBG-3 weakened it, RBG-4 confirmed inverse evidence. | [`results/v5_inverse_evidence/`](results/v5_inverse_evidence/README.md) |
| **V5-RBG-5** | Can a naturally successful internal state repair a failed prose/assertion choice? | Behavior replicated; mechanistic gate failed before activations. | [`results/v5_mechanistic_decomposition/`](results/v5_mechanistic_decomposition/README.md) |
| **Stage 2 — controlled composition** | Can the mechanism compose with a fact inferred from a story? | Not reached. | [`docs/literature-review.md`](docs/literature-review.md#stage-2-controlled-composition) |
| **Stage 3 — naturalistic behavior** | Does the mechanism matter in a richer strategic setting? | Not reached. | [`docs/literature-review.md`](docs/literature-review.md#stage-3-naturalistic-behavior) |

The first H0R-C corpus is also recorded because it was rejected before the
intervention was opened: [`results/v2_h0r_argument_validation/`](results/v2_h0r_argument_validation/README.md).
It is a **Stage 1E.4a corpus-eligibility check**, not a separate research stage.

The later V3–V5 studies are deliberately listed after the Stage 1 path. They
are protocol generations and behavioral follow-ons, not Stage 2 and Stage 3 of
the original mechanistic ladder.

## What the results mean

Three conclusions are supported by the committed evidence:

- The model can follow the simple reporting task, and the lens implementation passes its numerical and structural checks.
- The frozen J-space write has a real, selective, strength-dependent effect on answer scores. In plain language, it can push the model toward a chosen answer direction.
- That effect is not reliable enough to count as a validated counterfactual instrument: it rarely made the chosen answer the model’s most likely answer on the fresh test.


## A small glossary

- **Fact / world state:** the answer that is true in the toy setup, such as A or B.
- **Reporting policy:** the rule for turning the fact into a report, such as “say the fact” or “say the other option.”
- **J-space:** the intermediate coordinate system exposed by the Jacobian Lens.
- **Lens:** a fixed mathematical readout that translates an internal model state into token-related coordinates.
- **Intervention:** a temporary, targeted change to an internal model state during a forward pass.
- **Top-1:** the single token with the highest model probability. “Target top-1” means the requested counterfactual answer actually became the model’s most likely answer.
- **Log-odds gain:** how much the intervention changed the target answer’s score relative to the source answer. It is useful for measuring direction, but it does not guarantee a top-1 change.
- **Control:** a comparison intervention that should not produce the same effect, such as an identity or random-direction write.
- **Gate:** a predeclared pass/fail rule. A failed gate closes the next stage instead of being explained away afterward.

## Repository map

The folder READMEs are the human-facing map. The files below are the working
parts behind them.

| Area | What it contains |
|---|---|
| [`results/`](results/README.md) | Raw outputs, summaries, figures, and a README for each experiment. |
| [`docs/`](docs/README.md) | Research question, literature boundary, analysis plans, preregistrations, reports, and the decision log. |
| [`src/jspace_policy/`](src/jspace_policy/README.md) | Reusable dataset, analysis, lens-direction, intervention, and diagnostic code. |
| [`scripts/`](scripts/README.md) | Deterministic analysis and figure-generation scripts. |
| [`configs/`](configs/README.md) | Experiment inputs, locked controls, and frozen intervention settings. |
| [`tests/`](tests/README.md) | Small automated checks for the reusable code. |

## Reproduce the local checks

The GPU runs are already committed as evidence. The lightweight code checks
can be run locally:

```bash
uv sync --extra dev --extra modal
uv run pytest
```

The committed summary tables and figures can be regenerated from their raw
rows with:

```bash
uv run python scripts/analyze_v2_h0.py
uv run python scripts/analyze_h0r.py
uv run python scripts/analyze_stage1.py --phase locked
uv run python scripts/analyze_workspace_atlas.py --phase open
uv run python scripts/analyze_workspace_atlas.py --phase locked
```

The Modal entry points preserve the original GPU workflow. Existing locked
result directories should not be treated as fresh runs or retuned after the
fact.

## The deeper record

- [`docs/v2/final-report.md`](docs/v2/final-report.md) explains the original H0 failure.
- [`docs/v2/h0r-diagnostic-report.md`](docs/v2/h0r-diagnostic-report.md) explains how the recovery candidate was chosen.
- [`docs/v2/h0r-final-report.md`](docs/v2/h0r-final-report.md) gives the final causal-branch verdict.
- [`docs/v2/decision-log.md`](docs/v2/decision-log.md) records what changed, why it changed, what had already been seen, and what remained unopened.

## References

- Gurnee et al., [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/) and the [reference implementation](https://github.com/anthropics/jacobian-lens).
- Zou et al., [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405).
- Burns et al., [Discovering Latent Knowledge in Language Models Without Supervision](https://arxiv.org/abs/2212.03827).
- Levinstein and Herrmann, [Still No Lie Detector for Language Models](https://arxiv.org/abs/2307.00175).
- Wang, Zhang, and Sun, [When Thinking LLMs Lie](https://arxiv.org/abs/2506.04909).
