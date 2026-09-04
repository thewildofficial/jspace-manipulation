# Code and evidence audit

## Scope

Reviewed the repository at `eb23897f12bf902763f43f4772bdf8b0ebd3f5e3`:
89 Python files / 29,224 lines were inventoried. Detailed inspection focused on
current and historical research specifications and verdicts; game generators;
J-lens directions, coordinate edits and hook placement; the V5 capture, patch,
probe and readout paths; RBG-6 branching and analysis; and budget/workflow design.
This is not a claim that every historical runner has been executed or every
stored activation reanalyzed. The original RBG-6 analysis was reproduced exactly
from its committed raw data on CPU.

The original scientific objective was to separate fact representation from the
policy that maps a fact into a report. Later iterations moved toward behavioral
phenomena when the intervention instrument failed. That evolution is reasonable;
the organization obscures which claims survived it.

## A1. Candidate J-lens scores omit final normalization

**Confirmed implementation discrepancy; effect on real Qwen rankings unmeasured.**
In `modal_v5_mechanistic_decomposition.py::jspace_remote`, full-vocabulary rows use
`model.unembed(transported)`, whereas optimized candidate rows multiply
`transported @ lm_head.weight[required_ids].T` directly. The pinned upstream
`jlens/hf.py::HFModel.unembed` applies the model's final norm before the head and
also supports a final logit softcap. These are different readouts.

The parity check evaluates a zero vector. For zero-preserving RMSNorm and a
bias-free head, both branches return zero regardless of the learned norm gains.
It therefore cannot validate normalization parity. A nonuniform gain can change
candidate rankings; it is not necessarily a harmless common rescaling.

The new [CPU audit artifact](../../results/research_audit/snapshot_eb23897.json)
contains a synthetic counterexample: omitted normalization chooses candidate 0,
normalized scoring chooses candidate 1, and the zero-vector check passes.
This proves the check's blind spot, not that any particular archived Qwen row
changes rank.

**Correction implemented in this branch:** `lens_readout.py` normalizes using the
pinned adapter's real final norm, preserves head dtype/bias/softcap, and replaces
the candidate path. The parity gate now rejects zero-only inputs. CPU regression
tests cover nonuniform gains, multiple scales, BF16, bias and softcap. New output
metadata identifies `normalized_candidates_v2`. Real-checkpoint GPU validation
and recomputation of archived scores remain outstanding.

**Next validation:** the versioned readout must normalize once using the actual
model module, select head rows, apply bias/softcap if present, and compare against
full unembedding on nonzero random and recorded states across norm scales.
Use exact dtype/device conventions. Recompute from saved residuals where available;
record an erratum and new artifact, preserving the original file and causal verdict.
Do not use the old mixed-path 54.69% aggregate as a verdict on the J-lens.
Upstream source: https://github.com/anthropics/jacobian-lens/blob/581d398613e5602a5af361e1c34d3a92ea82ba8e/jlens/hf.py

## A2. Failed identity control limits causal interpretation

RBG-5B reports a maximum identity margin shift of 0.5, exceeding its required
1e-4 tolerance. The average primary patch margin gain is approximately 0.020.
Its 4% repair and p=.429 fail the locked endpoint independently.

`_capture_split` records states under capture batches. `_patched_query_batch`
recomputes recipient prompts under patch batches. `_run_locked_patches` compares
patched margins to earlier behavior outputs, rather than a contemporaneous
unpatched forward with identical batching. This introduces a plausible source of
numerical differences, especially for BF16/hybrid recurrent kernels. The code
inspection does **not** identify batching as the proven cause.

Next instrument validation must compare: no hook; a no-op hook; in-pass identity
replacement; and stored-state replay, all with the same batch shape, padding,
kernels and dtype. Compare current clean logits with archived clean logits as a
separate reproducibility diagnostic. A tolerance must be calibrated on neutral
controls before outcome inspection. Preserve ties and score strict repairs; a
zero-margin argmax tie is weak evidence.

The historical result supports 'this frozen intervention did not establish
selective repair,' not 'there is no causal mechanism in the model.'

## A3. RBG-6 establishes transcript dependence, with important confounds

`full_action_trajectory.py::trajectory_messages` ends with an instruction to use
just-reported consequences. Direct action lacks that instruction, the extra
turns, and their text. The rescue contrast changes all those features together.
Oracle-versus-swapped replay is a cleaner content contrast, but occurs under an
instruction explicitly privileging the reports. Strong swapped-report following
is not by itself evidence of spontaneous self-trust or endogenous deception.

`modal_full_action_trajectory.py::_query` takes legal-token argmax and renders the
chosen tokens into later turns. This is a legitimate constrained trajectory.
It is neither free-form chain-of-thought extraction nor an independent persistent
mental state surviving outside tokens. The full prefix is recomputed for each
query. Identical roles/tokens should give the same deterministic computation
subject to execution numerics; 'self-generated' and identical oracle replay are
not independent authorship treatments.

Thinking is disabled. Conclusions cannot be extended to the checkpoint's thinking
mode without a new test. Formatting compliance is recorded, which is good;
future runs should also distinguish legal-choice accuracy from unconstrained
first-token accuracy and failed generations.

## A4. RBG-6 uncertainty and prose need an addendum

CPU reproduction confirms the frozen analysis and raw SHA. The primary 48 rows
are two report orders for 24 source conditions/base games, using 12 distinct
unordered concept pairs. They are not 48 independent sampled games.

The recorded 0/48 exact binomial interval, 0–7.40%, assumes independent Bernoulli
trials. Treat it as the archived row-level calculation, not a cluster-valid
population bound. The distinct base-level outcome 'any dissociation across the
two orders' has 0/24 observations; a binomial interval for that *different*
estimand is 0–14.25%, still relying on independent sampled bases. Lexical reuse
limits broader generalization further. The paired rescue p-value already uses
base-game clusters and reproduces.

The findings' sentence saying the two report errors did not produce wrong actions
is contradicted by raw data: both report-error trajectories chose incorrectly.
The correct conditional result is **46/46 correct actions when both reports were
correct**, with two additional incorrect-report/incorrect-action trajectories.
The unconditional self-generated accuracy remains 46/48. This does not overturn
the rescue result; it clarifies its interpretation.

## A5. A token inventory is not a relational state

The V2 trajectory report already correctly distinguishes top-k J-lens readouts
from formal sparse nonnegative J-space decomposition. Preserve that distinction
throughout new work. Taking mean subtoken scores for a multi-token concept does
not validate a phrase representation. Neither the presence of 'lie' nor its
absence measures a deception policy without controlled contrasts.

The earlier H0R result includes substantial directional log-odds shifts despite
failing its top-1 threshold. V3 showed positive burned-set directionality but
stopped before fresh intervention. Treating all of this as 'J-space does nothing'
loses information; treating it as confirmed strategic control overstates it.

## A6. Compute and reproducibility

Useful infrastructure exists: deterministic oracles, frozen hashes, grouped
splits, overwrite refusal, workflow artifacts and budget utilities. Reuse those.

Remaining risks for new runs:

- Several historical GPU functions also tokenize, fit probes or run statistics.
  RBG-6's pre-rendered CPU branch enumeration is the better template.
- Default model forwards do not uniformly disable caches. For full-prefix scoring,
  explicitly test `use_cache=False`, `logits_to_keep=1` and no full hidden-state
  output. Capture only selected anchors with hooks.
- Cost ledgers written only after successful return omit failed runs unless
  reconciled. Reserve worst-case resources before launch, retain reservations
  after failure, reconcile with provider billing, and serialize stages.
- Broad remote dependency ranges do not provide exact environment reproduction.
  Pin the validated image and record attention implementation and package versions.
- Use length buckets and a measured token budget, rather than assuming a fixed
  batch of 24 is safe. BF16 27B parameters alone occupy about 54 GB before runtime
  overhead. No automatic paid retry after OOM.

The proposed work preserves historical evidence and introduces fresh versioned
implementations. No failed preregistration is reopened by this audit.
