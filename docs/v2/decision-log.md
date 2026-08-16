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

## 2026-08-17 — V2-D009 — Close H0R after prospective H0R-C failure

- **Change:** mark H0R-C failed, prohibit H0R-D, and keep the V2 causal branch
  blocked without creating a validated protocol artifact.
- **Reason:** the replacement baseline passed at 96.2%, and all gain,
  directional, selectivity, KL, displacement, conditioning, and cosine criteria
  passed, but target-answer top-1 was only 4/125 (3.2%) versus the frozen 20%
  minimum.
- **Already observed:** the one prospective H0R-C baseline and frozen semantic,
  random, unrelated, and direct-answer intervention results. No H0R-D behavior
  or strategic-reporting data was opened.
- **Confirmatory status:** H0R-C is a prospective failure. H0R-D is not run; the
  conjunction required to unblock V2 is false.
- **Files:** `results/v2_h0r_argument_validation_v2/`,
  `docs/v2/h0r-final-report.md`.
- **Commit:** recorded by the terminal H0R result commit.

## 2026-08-17 — V2-D010 — Preregister observational Stage 1

- **Change:** open a non-causal latent state–report dissociation study with
  explicit-state and inferred-state substages, neutral four-state report
  transformations, family-held-out splits, frozen pre-output J-space scores,
  independent residual probes, behavioral gates, and a once-opened locked test.
- **Reason:** H0R did not validate counterfactual replacement, but its failure
  does not answer the prior observational question of whether task state
  remains recoverable when report policy selects a different output.
- **Already observed:** all immutable Stage 0/H0R results and historical V1
  observational artifacts. No Stage 1 prompt tokenization, behavior, residual
  activation, J-space score, probe performance, or locked-family result has
  been opened.
- **Confirmatory status:** generator families/splits, model/lens identifiers,
  layers 36–43, final pre-output position, K/Q/M/D signs, exclusions,
  behavioral thresholds, probe grid/selection rule, bootstrap, and success
  criterion become confirmatory when this entry and associated code/config are
  committed.
- **Files:** `docs/v2/stage1-preregistration.md`,
  `docs/v2/stage1-analysis-plan.md`, `configs/v2/stage1.json`,
  `src/jspace_policy/stage1.py`, `src/jspace_policy/stage1_analysis.py`,
  `modal_stage1.py`, `scripts/analyze_stage1.py`, `tests/test_stage1.py`.
- **Commit:** recorded by the preregistration commit before tokenizer-only
  dataset freezing.

## 2026-08-17 — V2-D011 — Replace a tokenizer-invalid candidate label

- **Change:** replace `birch` with `tree` in the fifth four-label vocabulary.
- **Reason:** the first tokenizer-only freeze stopped because `birch` is not a
  single continuation token after the frozen answer prefix. The replacement is
  a common, semantically task-irrelevant label subject to the identical frozen
  tokenization check.
- **Already observed:** the tokenizer exception naming `birch`; no dataset was
  written and no model forward pass, behavior, activation, J-space score, or
  probe output was opened.
- **Confirmatory status:** unchanged. This is the preregistered mechanical
  tokenization rule applied before the rendered corpus is frozen.
- **Files:** `src/jspace_policy/stage1.py`.
- **Commit:** recorded with the tokenizer-enriched dataset freeze.

## 2026-08-17 — V2-D012 — Reject raw-prompt behavior and use pinned chat rendering

- **Change:** preserve the first rendered corpus and behavior run as invalid;
  render the unchanged task texts through Qwen's pinned chat template with
  `enable_thinking=False`; use left padding and request only final-position
  logits from Transformers.
- **Reason:** all 1,440 development rows returned the identical raw-model
  paragraph token `\\n\\n` as top-1, yielding 0% exact-format accuracy. The
  repository's established Qwen behavioral path uses the chat template. The
  run also logged an allocation warning while materializing unused
  full-sequence logits; final-only logits remove that infrastructure issue
  without changing any scientific score.
- **Already observed:** 0% exact top-1 accuracy in both substages; all top-1
  tokens were ID 271; candidate-restricted argmax accuracy was 89.3% for 1A and
  28.1% for 1B. No activation, residual probe, J-space, or locked-family output
  was opened.
- **Confirmatory status:** the first corpus/run is an invalid behavior-format
  gate and remains committed. Behavioral redesign is permitted before any
  mechanistic output. Families, evidence generators, states, codebooks, splits,
  thresholds, metrics, and mechanistic definitions are unchanged.
- **Files:** `modal_stage1.py`, preserved v1 dataset/behavior artifacts, and the
  replacement `configs/v2/stage1_dataset.json`.
- **Commit:** recorded before opening replacement behavior.
