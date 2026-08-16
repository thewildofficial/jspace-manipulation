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
