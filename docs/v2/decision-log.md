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

## 2026-08-17 — V2-D013 — Retry interrupted replacement behavior job

- **Change:** rerun the unchanged replacement discovery/validation behavior
  command after the local Modal client was interrupted and the remote app was
  stopped.
- **Reason:** the interrupted invocation returned no result and wrote no
  `behavior_dev.json`; there is therefore no partial scientific artifact to
  analyze or overwrite.
- **Already observed:** model-loading progress and Modal heartbeat/stop errors
  only. No replacement prompt output, aggregate behavior, activation, J-space,
  residual-probe, or locked-family result was materialized locally.
- **Confirmatory status:** unchanged. Dataset, prompts, runner, gates, and seeds
  are identical for the retry.
- **Files:** append-only decision log only.
- **Commit:** recorded before retrying replacement behavior.

## 2026-08-17 — V2-D014 — Simplify behaviorally invalid positive-control tasks

- **Change:** retain only the four vocabularies whose labels are one-token
  continuations without a leading-space fallback; remove the extra person-to-
  label indirection in Stage 1B; replace weak arithmetic, order-statistic,
  chain, and cycle families with one-step parity, primality, string-case, and
  set-intersection inference families. The three still-unopened locked families
  are specified as inventory maximum, eligibility elimination, and checksum
  matching.
- **Reason:** the valid chat-rendered development gate reached 87.8% in 1A and
  50.8% in 1B. Stage 1A was 100% for the first four vocabularies but 50% and
  75% for two vocabularies whose natural answer began with a subtoken rather
  than the frozen leading-space token. In 1B, direct selection families were
  strongest, while modular arithmetic, second-smallest, symbol-chain, and
  cycle-offset families were invalid positive controls.
- **Already observed:** discovery/validation behavior only, including family,
  vocabulary, state, and policy accuracies. No activation, J-space, residual-
  probe, or locked-family output was opened.
- **Confirmatory status:** behavioral redesign is permitted. The replacement
  keeps four states, neutral derangements, factorial grouping, family-level
  splits, sample size, thresholds, mechanistic scores, probe protocol, and
  bootstrap unchanged. Replaced development families are new and unopened.
- **Files:** `src/jspace_policy/stage1.py`, `modal_stage1.py`, preserved v2
  dataset/behavior artifacts, and replacement dataset.
- **Commit:** recorded before opening the third development behavior corpus.

## 2026-08-17 — V2-D015 — Retry third behavior corpus after connection failure

- **Change:** retry the unchanged third-corpus behavior command.
- **Reason:** the Modal client could not connect to the service; no remote app
  was created and no result file was written.
- **Already observed:** a client connection error only. No prompt output or
  scientific result was materialized.
- **Confirmatory status:** unchanged; the retry uses the same committed corpus,
  runner, model revision, and gates.
- **Files:** append-only decision log only.
- **Commit:** recorded before retry.

## 2026-08-17 — V2-D016 — Replace the invalid prime-classification family

- **Change:** replace only the discovery `unique_prime` family with
  `unique_negative`, in which exactly one of four label-associated values is
  negative.
- **Reason:** the third development gate passed Stage 1A at 100% and Stage 1B
  overall at 90.6%, but `unique_prime` reached only 55%, below the frozen 80%
  family minimum, and the Stage 1B x3/transformed cell reached only 77.8%, below
  85%. All other Stage 1B families were at least 88.8%.
- **Already observed:** third-corpus discovery/validation behavior only. No
  activation, J-space, residual-probe, or locked-family output was opened.
- **Confirmatory status:** behavioral redesign remains permitted. One unopened
  replacement discovery family is substituted; all other corpus and analysis
  choices are unchanged.
- **Files:** `src/jspace_policy/stage1.py`, preserved v3 dataset/behavior
  artifacts, and replacement dataset.
- **Commit:** recorded before opening the fourth development behavior corpus.

## 2026-08-17 — V2-D017 — Pass the development behavioral gate

- **Change:** declare the fourth frozen corpus behaviorally eligible for
  discovery/validation mechanistic execution.
- **Reason:** Stage 1A achieved 100.0% overall, cell, and family accuracy. Stage
  1B achieved 95.4% overall accuracy, with minimum state-policy cell 88.9% and
  minimum family 88.8%, exceeding the frozen 90%/85%/80% thresholds.
- **Already observed:** behavior-only outputs on discovery and validation
  families. Locked families remain unopened. No activation, J-space score, or
  residual-probe output has been observed.
- **Confirmatory status:** dataset design is now frozen. No further family,
  prompt, vocabulary, codebook, formatting, or exclusion redesign is permitted
  for confirmatory Stage 1.
- **Files:** `results/v2_stage1/raw/behavior_dev.json` and cost ledger.
- **Commit:** recorded before the first mechanistic run.

## 2026-08-17 — V2-D018 — Freeze the Stage 1 locked-test protocol

- **Change:** freeze the fitted discovery-only residual probes, validation-
  selected layers/regularization, exact locked families, run hashes, final
  analysis implementation, and claim ladder for a once-opened locked test.
- **Reason:** discovery/validation mechanistic execution completed under the
  preregistered band and generated the artifacts needed to apply independent
  probes prospectively.
- **Already observed:** all discovery/validation J-space, logit-lens, output,
  and residual-probe results. Stage 1A transformed band K was negative in every
  development family; Stage 1B transformed band K was slightly positive in
  eight of nine development families. No locked-family behavior or mechanistic
  output has been opened.
- **Confirmatory status:** the locked test is confirmatory and will be executed
  once. No prompt, family, layer, score, probe, threshold, exclusion,
  bootstrap, or claim rule may change based on its outcome.
- **Files:** `configs/v2/stage1_probe_freeze.json`,
  `docs/v2/stage1-freeze.md`, development raw results/summaries/figure, and the
  unit-tested shared candidate-evidence function.
- **Commit:** the freeze commit immediately preceding locked behavior.

## 2026-08-17 — V2-D019 — Pass the locked behavioral gate

- **Change:** declare both locked substages behaviorally valid and open their
  once-run mechanistic readout under the frozen protocol.
- **Reason:** locked Stage 1A achieved 100.0% overall, cell, and family accuracy.
  Locked Stage 1B achieved 98.75% overall, with minimum cell 96.7% and minimum
  family 96.25%; all frozen thresholds pass.
- **Already observed:** locked behavior only. No locked activation, J-space,
  logit-lens, or residual-probe output has been observed.
- **Confirmatory status:** both locked mechanistic tests are valid and will be
  opened once with the committed probe artifact and analysis protocol.
- **Files:** `results/v2_stage1/raw/behavior_locked.json` and cost ledger.
- **Commit:** recorded immediately before locked mechanistic execution.

## 2026-08-17 — V2-D020 — Close Stage 1 after the locked test

- **Change:** mark both frozen J-space retention criteria failed, report the
  independently positive residual-probe result, and keep Stage 2 closed.
- **Reason:** locked Stage 1A transformed band K was -0.108 (95% CI -0.142 to
  -0.073) with 0/3 positive families. Stage 1B was -0.017 (95% CI -0.043 to
  0.008) with 2/3 positive families. Although locked all-policy state probes
  reached 100.0% and 99.2%, positive K was a required conjunct.
- **Already observed:** the complete once-run locked behavior, J-space,
  logit-lens, output-logit, and residual-probe results.
- **Confirmatory status:** Stage 1 is a confirmatory negative for the pinned
  J-space retention criterion. The residual-stream decoding result is positive
  and supports only the narrower conclusion preregistered for this outcome.
  Stage 2 is not licensed.
- **Files:** `results/v2_stage1/`, `docs/v2/stage1-final-report.md`, root
  `README.md`.
- **Commit:** recorded by the final Stage 1 results commit.

## 2026-08-17 — V2-D021 — Preserve duplicate clusters in paired bootstraps

- **Change:** assign each resampled scenario occurrence a unique bootstrap
  instance before computing paired policy differences.
- **Reason:** final reproducibility QA found that duplicate scenario IDs were
  retained for mean-K and probe estimands but collapsed by the dictionary used
  for secondary paired report-preparation, state-policy, and interaction
  effects. The point estimates were correct; only their secondary intervals
  required regeneration.
- **Already observed:** all locked results. No model output is rerun and no
  endpoint, threshold, sign, exclusion, seed, or claim rule changes.
- **Confirmatory status:** headline transformed-K intervals, family directions,
  probe intervals, and both failed criterion decisions are unchanged. The
  repaired paired intervals are reported as the reproducible final values.
- **Files:** `src/jspace_policy/stage1_analysis.py`, regression test, regenerated
  summaries/report/figure.
- **Commit:** recorded in the final Stage 1 results commit.

## 2026-08-17 — V2-D022 — Open V2-E1 Strategic Workspace Atlas

- **Change:** open a separate observational exploratory study across six
  incomplete-information and information-control game families. Freeze the
  model/lens identifiers, deterministic solvers, dataset seed, rendering-level
  splits, behavioral gates, full-layer top-k readout, residual/J-space/output
  probe variables, deep-atlas rule, overwrite refusal, and cost ceiling before
  tokenization or model execution.
- **Reason:** Stage 1 found strong residual state decodability without positive
  state visibility under its four-candidate J-space endpoint. The new question
  is which independently calculated strategic variables become vocabulary-
  visible, rather than whether one privileged state token remains active.
- **Already observed:** all prior repository results; deterministic local game
  outputs and solver-validation tests only. No V2-E1 tokenizer result, model
  behavior, activation, J-lens token, residual-probe metric, or locked-
  replication output has been observed.
- **Confirmatory status:** discovery and validation are explicitly exploratory.
  The locked rendering split is reserved for hypotheses frozen only after open-
  atlas inspection. V2-E1 does not reopen H0/H0R or license causal claims.
- **Files:** `docs/v2/strategic-workspace-atlas-spec.md`,
  `configs/v2/workspace_atlas/experiment.json`,
  `src/jspace_policy/workspace_atlas.py`, `modal_workspace_atlas.py`,
  `scripts/analyze_workspace_atlas.py`, and `tests/test_workspace_atlas.py`.
- **Commit:** the V2-E1 protocol commit immediately before dataset tokenization.

## 2026-08-17 — V2-D023 — Freeze the V2-E1 tokenized corpus

- **Change:** freeze 216 chat-rendered rows across six games and three splits,
  with 108 paired bootstrap groups and one-token legal action IDs 32/33/34.
- **Reason:** all deterministic solver tests passed, the remote pinned tokenizer
  represented `A`, `B`, and `C` as distinct one-token continuations, and the
  rendered sequence lengths (74--147 tokens) fit the frozen execution design.
- **Already observed:** tokenizer output, content hash, row/group counts, token
  IDs, and sequence lengths only. No model forward pass, action probability,
  activation, J-lens readout, or probe result has been observed.
- **Confirmatory status:** discovery/validation remain exploratory; locked
  replication remains unopened. The corpus content hash is
  `6df46e0480d191b0f948e506710ccad7f347dba88a1e31a944eca7028635ba84`.
- **Files:** `configs/v2/workspace_atlas/dataset.json`.
- **Commit:** the dataset-freeze commit immediately before open behavior.

## 2026-08-17 — V2-D024 — Reject V2-E1 corpus v1 at the behavior gate

- **Change:** preserve the first open behavior corpus as invalid and freeze one
  behavior-only redesign before any mechanistic output. Remove tied optima,
  widen the small Kuhn expected-value margins, explicitly tell the model to use
  the supplied opponent policy, and show computed payoff rows in the inspection
  and disclosure calibration prompts.
- **Reason:** formatting was 100% and pooled exact-game accuracy was exactly
  70%, but Kuhn accuracy was 41.7%, below the unchanged 50% per-family gate.
  Errors were concentrated in check decisions with small value margins; the
  model selected `BET` on all 24 Kuhn rows. Inspection also showed a large
  rendering effect, and full-disclosure rows were systematically missed. The
  redesign makes the intended expected-utility task unambiguous rather than
  lowering a threshold or selecting a mechanistic result.
- **Already observed:** open behavior only: family accuracies, legal-action
  distributions, output probabilities, and solver regret. No activation,
  J-lens readout, logit-lens trajectory, residual probe, or locked behavior has
  been observed.
- **Confirmatory status:** V2-E1 remains exploratory and the behavioral
  thresholds are unchanged. Corpus v1 and its failed behavior report remain
  preserved. The locked split remains behaviorally and mechanistically
  unopened.
- **Files:** `configs/v2/workspace_atlas/dataset_v1_failed_gate.json`,
  `results/v2_workspace_atlas/raw/behavior_open_v1_failed_gate.json`, updated
  generator/tests/config, and the retained v1 summaries/figure/report.
- **Commit:** the redesign commit immediately before v2 tokenization.

## 2026-08-17 — V2-D025 — Freeze V2-E1 corpus v2

- **Change:** freeze the behaviorally redesigned 216-row corpus with the same
  games, split sizes, paired groups, and legal action IDs as v1.
- **Reason:** all solver tests pass after the redesign; every row now has a
  unique payoff-maximizing action, action diversity remains present in every
  game, legal tokens remain IDs 32/33/34, and rendered lengths are 74--166
  tokens.
- **Already observed:** v1 behavior and v2 tokenizer/solver metadata only. No v2
  behavior, activation, J-lens, logit-lens, residual-probe, or locked output has
  been observed.
- **Confirmatory status:** thresholds and mechanistic protocol remain unchanged.
  The v2 corpus hash is
  `e37873b3a44e29aeb96798aca6a38bcd04b1b99656bd66214b787df33874df24`.
- **Files:** `configs/v2/workspace_atlas/dataset.json`.
- **Commit:** the v2 dataset-freeze commit immediately before its open behavior.

## 2026-08-17 — V2-D026 — Pass the V2-E1 open behavioral gate

- **Change:** declare corpus v2 behaviorally eligible for open-atlas
  mechanistic execution under the frozen all-layer protocol.
- **Reason:** all 144 outputs were formatting-compliant; pooled exact-game
  optimal accuracy was 78.3% versus the 70% floor; and the weakest exact game
  reached 62.5% versus the 50% family floor. Pooled family accuracies were
  inspection 91.7%, Kuhn 62.5%, cheap talk 62.5%, signaling 83.3%, disclosure
  91.7%, and controlled Chameleon 62.5% (descriptive, not gated).
- **Already observed:** open behavior and solver regret only. Validation cheap-
  talk behavior was weaker than discovery behavior and will be retained rather
  than excluded. No activation, J-lens, logit-lens trajectory, residual-probe,
  or locked output has been observed.
- **Confirmatory status:** the open mechanistic atlas is exploratory. Dataset,
  game solvers, thresholds, layers, positions, top-k rules, variables, and
  probe training/evaluation splits are now frozen for this run.
- **Files:** `results/v2_workspace_atlas/raw/behavior_open.json`, behavior
  summaries/figure/report, and the cumulative cost ledger.
- **Commit:** the behavior-gate commit immediately before open mechanistic
  execution.

## 2026-08-17 — V2-D027 — Freeze V2-E1 locked replication endpoints

- **Change:** after inspecting the open atlas, freeze four separate locked
  endpoints: layer-43 generic optimization-token visibility in Kuhn/signaling;
  a five-game residual-over-J-space/output strategy-decoding contrast; same-
  action Kuhn bluff/thin/value decoding; and late Kuhn/signaling action
  commitment. Add locked-only decoder training/evaluation support without
  changing the underlying readout.
- **Reason:** the open atlas showed `optimal`/`optimize` variants in every Kuhn
  and signaling row at layer 43, while their median action commitment occurred
  at layers 62 and 60.5. Strategy class was strongly validation-decodable from
  residual activations in the five exact games, but top-50 J-space rank features
  and output logits were generally much weaker. The Chameleon `safe`/`risk`
  pattern was not advanced because it failed open-template transfer.
- **Already observed:** all open behavior, 144 all-layer final-position
  readouts, 12 deterministic deep traces, and 4,572 open decoder evaluations.
  No locked behavior, activation, J-lens token, probe metric, or endpoint has
  been observed.
- **Confirmatory status:** every selection above is exploratory on the open
  atlas and prospective only on the locked rendering. Exact layers, token
  family, classifiers, thresholds, censoring, endpoint rules, and claim limits
  are frozen in the machine and prose replication documents. The locked
  behavior gate remains unchanged and is run before any locked activation.
- **Files:** `configs/v2/workspace_atlas/replication_freeze.json`,
  `docs/v2/strategic-workspace-atlas-freeze.md`, open raw/summaries/atlas/
  figures/report, and locked evaluation support in the runner/analysis script.
- **Commit:** the replication-freeze commit immediately before locked behavior.

## 2026-08-17 — V2-D028 — Pass the V2-E1 locked behavioral gate

- **Change:** declare the 72-row locked rendering behaviorally valid and open
  the once-run locked mechanistic replication.
- **Reason:** formatting was 100%, pooled exact-game accuracy was 85.0% versus
  the 70% floor, and the weakest exact game was Kuhn at 66.7% versus the 50%
  family floor. Other exact-game accuracies were cheap talk 83.3%, signaling
  75.0%, inspection 100%, and disclosure 100%. Chameleon reached 25% but is
  explicitly outside the exact-game gate and frozen quantitative endpoints.
- **Already observed:** all open atlas evidence and locked behavior only. No
  locked activation, J-lens readout, probe metric, commitment trajectory, or
  endpoint result has been observed.
- **Confirmatory status:** H1--H4 are now behaviorally valid for their named
  exact games and will be opened once under the committed freeze. Chameleon is
  retained as a descriptive negative transfer result.
- **Files:** `results/v2_workspace_atlas/raw/behavior_locked.json`, locked
  behavior summaries/figure/report, and cumulative cost ledger.
- **Commit:** the locked behavior-gate commit immediately before locked
  mechanistic execution.
