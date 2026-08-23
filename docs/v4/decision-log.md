# V4 append-only decision log

This file records what was known when each decision was made. Later corrections
are appended rather than silently rewriting the original rationale. Git history
contains the corresponding frozen protocols and raw artifacts.

## 2026-08-23 — Calibration V1 failed parsing

- Run: `859fd3da78374e3b8118c1ef8659aeca`.
- Both generated-rationale variants had zero parseability because the generation
  format did not reliably terminate in the required final line.
- Decision: do not inspect mechanisms; redesign only the behavioral formatting
  scaffold.

## 2026-08-23 — Calibration V2 hit the token ceiling

- Run: `7b1110ec4f3b412a81f6ce5a68236d64`.
- Parseability was 71.25% and unconditional accuracy 68.75% at the 96-token cap.
- Inspection showed 55/57 completed trajectories were correct; the competence
  problem was primarily truncation.
- Decision: raise the bounded rationale allowance to 160 tokens without changing
  the underlying games or accuracy thresholds.

## 2026-08-23 — Calibration V3 passed

- Run: `51097dab64ca40d3a18017dd3863ea7d`.
- Parseability was 100%; overall, strategic, and nonagentic accuracy were each
  97.5%.
- Decision: license the frozen 240-decision behavioral screen.

## 2026-08-23 — Behavioral screen passed

- Run: `7bb82d26fe0d4383be7ef53ea7a1dc38`.
- Accuracy was 234/240 (97.5%) with no token-ceiling hits.
- This established task competence only. It did not establish receiver modeling,
  theory of mind, or faithfulness of the rationale.
- Decision: license the discovery report fork; no intervention was licensed.

## 2026-08-23 — Initial reports exposed response-label contamination

- The original A/B/C report labels overlapped with the A/B/C decision labels.
- Predicted-response report accuracy in discovery was 44.9% with the rationale
  and 69.2% answer-only. Repetition of the selected action label was a major
  alternative explanation.
- Decision: do not claim a self-report gap. Freeze a disjoint X/Y/Z report
  replication before further interpretation.

## 2026-08-23 — Disjoint-label discovery found a rationale penalty

- Run: `2ae0cb48e3c24cc5bda87eefa60d507b`.
- On 78 correct discovery decisions, predicted-response accuracy was 70/78 with
  the rationale and 77/78 answer-only; discordance was 0 versus 7, exact
  p=.015625. This was one of four frozen controls; Holm p=.0625.
- All seven answer-only-only pairs followed selected action C.
- A matched unrelated trajectory was worse than the model's own rationale, so the
  result did not identify a uniquely self-generated effect.
- Decision: freeze a minimal held-out C-decision confirmation using only
  validation and locked splits.

## 2026-08-23 — Held-out C confirmation passed

- Run: `acbb2031038d4573a88d3232cb39322b`.
- On 54 correct held-out C decisions, accuracy was 47/54 with the rationale and
  53/54 answer-only; discordance was 0 versus 6, exact p=.03125.
- Manual inspection found correct arithmetic and the correct high-probability
  response explicitly present in the important discordant rationales.
- Decision: accept a narrow trajectory-interference phenomenon and freeze a 2x2
  response-alias discriminator. No mechanistic sweep was licensed.

## 2026-08-23 — Report-only aliases did not rescue retrieval

- Run: `56f00942b4f24b0e91d2f8c2a023ca44`.
- Pooled indexed accuracy was 75/88 rationale versus 87/88 answer-only. With
  Kestrel/Lumen/Quartz aliases it was 69/88 versus 85/88.
- Initial interpretation: aliases did not support the narrow C-to-R3 collision
  account and instead favored broader relational interference.
- Decision at the time: design a targeted internal-state study.

## 2026-08-23 — Correction to the alias interpretation

- The alias manipulation changed only the report mapping. The decision prompt and
  rationale retained A/B/C and R1/R2/R3.
- Therefore an ordinal association formed during the decision could survive the
  alias manipulation. Aliases also added a second mapping burden.
- Corrected decision: the alias result supports interference but does not localize
  it away from action/response ordinal binding. Freeze a prospective permutation
  that independently changes labels and presentation positions upstream.

## 2026-08-23 — V4-SES-2 ordinal study frozen before execution

- Config: `configs/v4/strategic_epistemic_search/ordinal_binding_permutation.json`.
- Dataset: 20 new base games × 9 OA variants × 2 matched frames = 360 decisions.
- Frozen dataset SHA-256:
  `6c23f1c0f4d2fe85e2be7a341bfcd485de6621d228545c0a6f78fb66bfafb318`.
- Independent factors: action-label assignment, action presentation position,
  response-label assignment, and response presentation position.
- Primary endpoints: label-lure and position-lure logit pull relative to the true
  response, averaged within base-game clusters; exact sign-flip tests with Holm
  correction.
- Decision: behavior gate first; report only if competence passes; stop before any
  mechanistic execution.

## 2026-08-23 — Ordinal behavior gate and embedded report completed

- GitHub Actions run: `32661361058`; artifact: `9499003037`.
- Artifact ZIP SHA-256:
  `106f78dd232e4dda8bc4042259f2677f37213b61e954d714a9fc87503bce0e0e`.
- Modal run: `02c5739b615b4cf9911c26c9ff7f7f53`.
- Source commit: `db454aa6e28848fe8538a1db7d7d530c2b817b42`.
- Behavior passed: 360/360 parseable, 345/360 correct, minimum cell 83.33%.
- Measured cost: $0.65315; buffered cost: $0.78378.
- Cumulative V4 measured cost: $2.08010 across nine GPU runs; cumulative buffered
  accounting: $2.49612. This remains far below the $18.50 protocol ceiling and
  the user's approximately $22 total allowance.

### Protocol deviation

The shared `behavior_remote()` function generated the 720 report rows during the
same Modal invocation even though the workflow named this the behavior stage. The
separate `ordinal-report` step was skipped. This was discovered only after the
immutable artifact was retrieved.

The competence gate passed and every report prompt, label randomization, endpoint,
and test was frozen before execution. No result-dependent prompt selection
occurred. The embedded report is therefore retained as the prospective report, but
the failure of stage separation is disclosed. The raw behavior payload remains
unchanged; `ordinal_binding_report_v1_from_behavior.json` is a lossless extraction
linked to source SHA-256
`7331a6b0f8fc594388c0605142f7db07f617c29e0fcc9051e0f57a7eb6d55a0c`.

The code now requires behavior entrypoints to pass
`include_self_report=False`. Future reports must use the separate report entrypoint.

## 2026-08-24 — Frozen ordinal tests passed but did not identify binding

- Report accuracy on 345 correct decisions: 332/345 rationale versus 342/345
  answer-only.
- Paired cells: 331 both correct, 11 answer-only-only, one rationale-only, two
  both wrong. Secondary unclustered exact p=.00635.
- Label-lure mean pull: +1.2177, raw exact p=1.91e-6, Holm p=3.81e-6.
- Position-lure mean pull: +1.2190, raw exact p=1.91e-6, Holm p=3.81e-6.
- Under the frozen lookup table this is "both / mixed binding code."

The two primary tests share a non-specific denominator: the true response. A
generic suppression of the truth relative to both false responses makes both tests
positive. Their almost identical effects triggered an explicitly post-hoc
specificity audit.

## 2026-08-24 — Post-hoc audit downgraded the binding claim

- Generic false-vs-true pull: +1.1097; 20/20 clusters positive; exact p=1.91e-6.
- Label lure versus the other false response: +0.1766; raw p=.0954; exploratory
  Holm p=.1908.
- Position lure versus the other false response: +0.1459; raw p=.2902;
  exploratory Holm p=.2902.
- Decision: do not claim label binding, position binding, or a mixed binding code;
  do not launch a broad mechanistic sweep from the frozen primary p-values.

The audit exposed a new exploratory interaction:

- nine of 11 answer-only-only errors were strategic versus two nonagentic;
- the strategic-minus-nonagentic generic false-vs-true pull was +0.6970, with
  19/20 clusters positive and exact p=3.81e-6; and
- the corresponding accuracy interaction was -4.44 points, exact p=.0547.

Decision: the cheapest next study is a fresh, specificity-aware confirmation of
the strategic-frame interaction using minimally different wording and multiple
paraphrase blocks. Only a successful confirmation licenses a narrow causal
residual/J-space localization and repair test.
