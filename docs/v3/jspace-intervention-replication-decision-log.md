# V3-JI1 decision log

## 2026-08-21 -- JI1-D001 -- Open a new intervention-characterization study

- Preserve the prospective H0R-C failure and its 20% top-1 criterion.
- Treat prior H0/H0R and released flexible-generalization outcomes as burned.
- Test native-raw alpha 0.5/1/2 swaps on a new behavior-only-frozen corpus.
- Replace a single conjunctive study verdict with directional, distortion, and
  behavioral-substitution evidence tiers.

## 2026-08-21 -- JI1-D002 -- Close design degrees of freedom

- Use first-feasible ordered corpus selection with no manual post-run repairs.
- Use per-hook/per-position delta-L2-matched deterministic random controls.
- Use balanced cyclic unrelated-semantic controls.
- Test both paired control contrasts rather than selecting a strongest control.
- Retain paper-style loading and require final-position-only robustness before
  interpreting the relation as persistent workspace recruitment.
- Exclude identity rows from rank/conditioning summaries.
- Describe historical distortion thresholds as low distortion on average and
  report distribution tails.

No new V3 intervention result had been generated or inspected when these
decisions were recorded.

## 2026-08-21 -- JI1-D003 -- Invalidate first Phase B launch before intervention

- The first burned launch entered dataset preparation but stopped before
  generating an intervention row because the new compatibility helper required
  the leading-space token form. Burned argument `lion` is single-token only in
  the standalone form under the pinned tokenizer.
- Restore the already-validated H0/H0R rule: prefer a one-token leading-space
  form, then accept a one-token standalone form.
- No V3 intervention result or summary artifact existed when this fix was made.

## 2026-08-21 -- JI1-D004 -- Separate burned answer eligibility from directions

- The second burned launch stopped before intervention because direction
  precomputation unnecessarily requested an answer token for every argument.
  Burned answer `savanna` has no suitable one-token form on the pinned tokenizer.
- Precompute only argument-concept directions. Apply the preregistered
  tokenization-permits rule to source/target answer scoring and record every
  excluded burned cell or target in the raw artifact.
- The stopped launch wrote no local result artifact and exposed no intervention
  row or summary.

## 2026-08-21 -- JI1-D005 -- Shard transport after client heartbeat failure

- A full burned run was cancelled after the local Modal client lost its
  heartbeat. Modal killed the worker when the client disconnected. The remote
  payload never returned, no local artifact was written, and no intervention
  outcome was inspected.
- Run each frozen category as a sealed shard. Each shard retains the complete
  condition matrix and reruns Phase A. A deterministic local combiner requires
  exact agreement on git commit, experiment hash, full-dataset hash, and category
  coverage before producing the preregistered phase artifact.
- This is an operational transport change only. Dataset contents, intervention
  conditions, controls, estimands, thresholds, and analysis remain unchanged.

## 2026-08-21 -- JI1-D006 -- Lazy-load Torch in shared analysis helpers

- The burned analyzer initially stopped before producing a summary because the
  local analysis environment does not install Torch and the shared pure helper
  module imported it eagerly.
- Move Torch imports into GPU/numerical functions. Dataset hashing, bootstrap,
  selection, recorded intervention rows, estimands, and gates are unchanged.
- The same first burned-analysis attempt then stopped while serializing a NumPy
  boolean from the frozen power calculation. Convert NumPy scalars to native JSON
  types; no statistic or decision rule changes.

## 2026-08-21 -- JI1-D007 -- Amend candidates after behavior-only v1 failure

- Candidate v1 produced 77/90 tokenization-eligible and 36/90 clean-correct
  cells. The deterministic selector found no feasible countries 4x4 block and
  refused to freeze. No fresh intervention had been generated.
- Preserve the v1 baseline artifact. Candidate v2 uses exact upstream-style
  country prompts, adds Australia for a fourth distinct one-token continent,
  replaces weak month facts with fixed month offsets, and replaces weak number
  templates with simple successor/predecessor and multiplication forms.
- The selector, tokenizer rules, requirement that every retained clean prompt be
  top-1 correct, intervention configuration, outcomes, and thresholds do not
  change. Candidate v2 is frozen before its clean-only run.

## 2026-08-21 -- JI1-D008 -- Amend candidates after behavior-only v2 failure

- Candidate v2 produced 89/90 tokenization-eligible and 60/90 clean-correct
  cells, but still had no feasible countries 4x4 block; month offsets also left
  only two fully reliable functions. No fresh intervention had been generated.
- Historical review additionally showed that several v2 country arguments had
  prior H0/H0R intervention exposure. Candidate v3 replaces them with previously
  unintervened Peru, Norway, Thailand, and Ghana (plus ordered fallbacks).
- Express two-step month functions compositionally using the already reliable
  immediate-next/immediate-previous wording. Retain the v2 number-word pool,
  whose five functions were clean-correct for all six arguments.
- Preserve v2's clean-only artifact. All selection, intervention, analysis, and
  evidence rules remain frozen.

## 2026-08-21 -- JI1-D009 -- Final candidate amendment after v3 failure

- Candidate v3 produced 76/90 tokenization-eligible and 56/90 clean-correct
  cells, but country tokenization remained sparse and month offsets continued to
  elicit punctuation. The unchanged selector again refused to freeze.
- Replace the country family with previously unintervened lowercase alphabet
  letters and five deterministic letter functions. Add an explicit month-name
  response instruction to the multi-step month functions. Retain number words.
- Candidate v4 is the final redesign attempt. If it lacks three feasible 4x4
  blocks, stop the study rather than weakening the behavior/tokenization gate.
- No fresh intervention has been generated. All causal and analysis parameters
  remain unchanged.
