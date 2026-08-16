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
