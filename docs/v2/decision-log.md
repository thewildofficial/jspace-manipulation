# V2 append-only decision log

## 2026-08-17 — V2-D001 — Execute H0 before strategic tasks

- **Change:** added a separate V2 pipeline and research trail; no V1 result was
  rewritten.
- **Reason:** the brief requires numerical, synthetic, directional, workspace,
  and flexible-generalization validation before safety interpretation.
- **Already observed:** all committed V1 artifacts plus ignored local V1 causal
  outputs described in the preregistration.
- **Confirmatory status:** H0 thresholds and the workspace selection rule are
  confirmatory once this entry is committed; later hypotheses remain gated.
- **Files:** `modal_v2.py`, `configs/v2/*`, `docs/v2/*`,
  `src/jspace_policy/interventions.py`, `scripts/analyze_v2_h0.py`.
- **Commit:** to be filled by the preregistration commit itself; the run
  manifest records the exact SHA.

## 2026-08-17 — V2-D002 — Serialize the Modal return as plain JSON

- **Change:** cast the remote PyTorch version object to text and serialize the
  complete remote return to JSON before transport.
- **Reason:** the first H0 invocation completed remotely but local
  deserialization failed because the local Modal environment has no `torch`
  module. No scientific result was downloaded or inspected.
- **Already observed:** model/lens load progress and the local deserialization
  traceback only; no H0 metric or gate value.
- **Confirmatory status:** unchanged. The workspace rule, prompts,
  interventions, metrics, exclusions, and pass thresholds are untouched.
- **Files:** `modal_v2.py`, `docs/v2/decision-log.md`.
- **Commit:** recorded by the subsequent run manifest.

## 2026-08-17 — V2-D003 — Cast the stored selection flag for plotting

- **Change:** cast the JSONL-derived `selected` column to boolean when filtering
  the deterministic workspace figure.
- **Reason:** pandas inferred JSON booleans as numeric values after grouping,
  causing the first post-run plotting command to stop before writing figures.
- **Already observed:** the complete immutable H0 raw output and gate result.
- **Confirmatory status:** unchanged; this affects only figure generation, not
  any metric, interval, threshold, exclusion, or gate decision.
- **Files:** `scripts/analyze_v2_h0.py`, `docs/v2/decision-log.md`.
- **Commit:** recorded in the final documentation commit.

## 2026-08-17 — V2-D004 — Open H0R after failed full-band causal control

- **Change:** open a separate causal-instrument recovery study; preserve H0 as
  failed and separate implementation parity from Qwen topology recovery.
- **Reason:** full-band/all-position swaps produced 0/90 target top-1 outcomes,
  while one-layer/all-position produced 25/90 overall and 24/33 for countries.
- **Already observed:** those values, the L36–43 operational band, center L40,
  and all numerical H0 parity checks. The Anthropic control set is burned.
- **Confirmatory status:** H0 remains failed. H0R-B is exploratory. Only the
  unopened H0R-C/D controls can prospectively unblock the causal branch.
- **Files:** `docs/v2/h0r-preregistration.md`,
  `docs/v2/h0r-analysis-plan.md`, `docs/v2/h0r-final-report.md`.
- **Commit:** recorded by Git history before locked-control construction.

## 2026-08-17 — V2-D005 — Replace LCM with a parity intermediate before locking

- **Change:** use computed sum parity (`even/odd`) with a fixed `RED/BLUE`
  downstream mapping for Control B instead of numeric LCM/GCD outputs.
- **Reason:** a tokenizer-only Modal pass found 0/30 LCM candidates with both
  natural and counterfactual answers as one-token continuations. The parity
  family retains computed-intermediate composition while using tokenizer-valid
  semantic states and reports.
- **Already observed:** tokenization eligibility only; no model forward pass,
  baseline behavior, J-lens readout, or intervention outcome.
- **Confirmatory status:** unchanged. The replacement is finalized and committed
  before H0R-B topology discovery, and will remain unopened until H0R-D.
- **Files:** `src/jspace_policy/h0r.py`,
  `configs/v2/h0r_locked_controls.json`, `docs/v2/decision-log.md`.
- **Commit:** recorded by the locked-control commit.

## 2026-08-17 — V2-D006 — Freeze the H0R candidate protocol

- **Change:** freeze layers 36--42, argument-through-end positions,
  pseudoinverse coordinate writing, alpha 0.5, and unit-normalized per-layer
  directions for both prospective controls.
- **Reason:** this configuration won the preregistered mean-rank rule after all
  conditioning, KL, displacement, function-stability, and direction filters.
- **Already observed:** all three burned-country diagnostic runs, including the
  layer, mask, strength, cumulative-set, semantic, control, and reconstruction
  results. No locked-control baseline or intervention result was opened.
- **Confirmatory status:** H0R-B is exploratory. The candidate becomes immutable
  for H0R-C and, only if C passes, H0R-D.
- **Files:** `configs/v2/h0r_candidate_protocol.json`,
  `results/v2_h0r_diagnostic/`, `docs/v2/h0r-diagnostic-report.md`.
- **Commit:** recorded by the candidate-freeze commit before H0R-C.

## 2026-08-17 — V2-D007 — Reject the first H0R-C corpus at baseline

- **Change:** declare the first 102-trial fresh argument corpus behaviorally
  invalid and permit one simpler replacement corpus under the existing
  preregistered redesign rule.
- **Reason:** baseline accuracy was 60.8%, below the frozen 80% minimum.
- **Already observed:** baseline top-1 correctness only: countries 86.7%; Greek
  sequence, chemical element, and weekday families 50% each. No intervention
  inference was executed (`intervention_opened: false`).
- **Confirmatory status:** the candidate protocol and all causal thresholds are
  unchanged and remain unopened on fresh argument outcomes. The replacement
  may use only baseline competence to simplify prompts/functions.
- **Files:** `results/v2_h0r_argument_validation/`,
  `docs/v2/h0r-final-report.md`.
- **Commit:** recorded before constructing the replacement corpus.

## 2026-08-17 — V2-D008 — Freeze replacement H0R-C corpus

- **Change:** freeze a 130-trial replacement argument corpus across easy
  countries, Greek successors, chemical symbols, and weekday successors.
- **Reason:** these functions were baseline-correct in the invalid first
  control; expanding only those easy families raises competence without using
  any intervention outcome.
- **Already observed:** first-control baseline correctness by function and
  tokenizer eligibility for the replacement. No replacement model forward pass
  and no fresh intervention outcome was observed.
- **Confirmatory status:** candidate protocol, 80% baseline gate, causal pass
  thresholds, and matched controls remain unchanged. The replacement content
  hash is `7f044c700fe16bb404cfafce90da47fa42fe1974997e3385ade243f819e2ce45`.
- **Files:** `configs/v2/h0r_locked_controls_v2.json`,
  `src/jspace_policy/h0r.py`, `modal_h0r.py`.
- **Commit:** recorded before the replacement baseline run.
