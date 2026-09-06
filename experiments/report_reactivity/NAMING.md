# Naming: one stable protocol ID

Report-reactivity experiments use **one stable descriptive protocol ID**. Do not
add alias layers for the same experiment.

## Rules

1. **Protocol / task ID** — the sole scientific identity. Used in protocol JSON
   (`protocol_id`), prepare `--task`, Actions workflow `task`, payload
   `protocol` / `protocol_id`, and `study_id` (same string; no parallel STUDY
   code).
2. **`run_id`** — results folder name under `results/report_reactivity/<run_id>/`
   only (e.g. `ask-mid-traj-qwen38-n16-v1`). Never treat a run_id as the
   experiment name in methods or claim text.
3. **Claim rows (`C##`)** — ledger indices, not experiment names.
4. **Markdown path kebab-case** (`ask-mid-trajectory-protocol.md`) — display /
   filesystem convenience only; docs must state the canonical ID in snake_case.
5. **Never put operator nicknames, queue numbers, or priority slogans** (PriGo,
   `#6`, `#7`, etc.) in scientific documentation. Those belong only in
   `docs/next-sprint/decision-log.md` (ops), clearly labeled.

## Live experiments

| Canonical protocol ID | Role | Results `run_id` (if scored) | Legacy / path aliases (do not use as IDs) |
|---|---|---|---|
| `ask_mid_trajectory` | Mid-trajectory ask-as-intervention | `ask-mid-traj-qwen38-n16-v1` | kebab markdown paths; historical study label `ASK-MID-TRAJECTORY-1` (pre-coherence; treated as synonym of the protocol ID) |
| `rename_invariant` | Rename vs consequence tool/button check | _(none yet — CPU prepare only)_ | kebab markdown paths; historical study label `RENAME-INVARIANT-1` |

Committed results directories are **not** renamed (manifests / hashes stay
immutable). Document them as `run_id`s under the stable protocol ID.

## Paid scoring gates (related)

- Paid GPU (`dry_run=false`) is blocked until the reporting path exists
  (`RENAME_GPU_SCORING_UNLOCKED` in `jspace_policy.report_reactivity_gates`).
  When unlocked later, require `pinned_payload_sha256` matching the reviewed
  prepare hash. See README + workflow comments.
