# Report-reactivity experiments

CPU prepare lives here (`prepare.py`, `prepare_preflight.py`, `protocol.json`).
GPU scoring is the root Modal entrypoint `modal_report_reactivity.py`.

**Naming:** one stable protocol ID per experiment — see [NAMING.md](NAMING.md).
`run_id` is the results folder only; claim rows (`C##`) are ledger indices.

Dispatch via `.github/workflows/report-reactivity.yml` (`workflow_dispatch`).
`dry_run` defaults to true so Actions can validate and materialize payloads
without Modal GPU spend. Set `dry_run=false` only for an intentional scoring
run, and pass `pinned_payload_sha256` equal to the reviewed prepare payload
hash (fail closed on mismatch). Default `batch_size` is 4 (see claim ledger C1).

`task=rename_invariant` with `dry_run=false` is **refused** until a complete
analyzer + results layout exists
(`jspace_policy.report_reactivity_gates.RENAME_GPU_SCORING_UNLOCKED`). CPU
prepare / dry_run remains allowed.

## Instrumentation confirmation (C11 / C12 / C13)

See [gha-cpu-dryrun-methods.md](gha-cpu-dryrun-methods.md):

- **C11:** Actions CPU dry-run `gha-dryrun-preflight-38-v1` confirmed.
- **C12:** Actions GPU `gha-preflight-38-v1` failed at ledger `reserve()`
  (stage cap on historical ledger); no GPU scores. Fix: GHA ledger
  `reservations_gha.jsonl` via `REPORT_REACTIVITY_LEDGER` ([RR-D001](../../docs/next-sprint/decision-log.md)).
- **C13:** Actions GPU `gha-preflight-38-v2` reserved on the GHA ledger and
  finished remote scoring, then failed locally unpickling torch; reservation
  retained; no `raw.json`. Fix: `score_gpu` returns JSON via `dumps_jsonable`
  (do not install torch on the Actions CPU runner).

## Discovery baseline on GHA (C14 / C15 / C16)

See [gha-report16-38-v1-methods.md](gha-report16-38-v1-methods.md):

- **C14:** Actions GPU `gha-report16-38-v1` (16-base discovery report baseline)
  completed with parity pass; ceiling on Direct/self/control/oracle/external;
  swapped accuracy 0.828125; primary contrasts 0.0. Discovery pilot only.
- **C15:** Same run — swapped prose 0.71875 vs opaque 0.9375 (descriptive).
- **C16:** Poisoned self-talk cell map — lied transcripts hurt most under wordy
  opposed strategic framing; see
  [poisoned-self-talk-cell-map.md](poisoned-self-talk-cell-map.md).
- Artifacts: `results/report_reactivity/gha-report16-38-v1/`; CPU analysis via
  `analyze_gha_report16.py`. No new GPU from documentation commits.

## Harder games (break Direct ceiling)

Protocol note:
[harder-games-break-the-ceiling.md](harder-games-break-the-ceiling.md).

Prepare / Actions knob: `--history-mode minimal|redundant` (workflow input
`history_mode`, default `minimal`). **Scored outcome**
(`harder-games-qwen38-n16-v1`, Actions 34051102729): redundant demos did **not**
break Direct ceiling (C17); swapped still leaks (C18). Methods:
[harder-games-qwen38-n16-v1-methods.md](harder-games-qwen38-n16-v1-methods.md).

## Mid-trajectory ask as intervention (canonical ID: `ask_mid_trajectory`)

Protocol:
[ask-mid-trajectory-protocol.md](ask-mid-trajectory-protocol.md) +
[protocol_ask_mid_trajectory.json](protocol_ask_mid_trajectory.json).

Canonical protocol / task / study ID: **`ask_mid_trajectory`**. Scored
`run_id` (results folder only):
[`ask-mid-traj-qwen38-n16-v1`](../../results/report_reactivity/ask-mid-traj-qwen38-n16-v1/)
(Actions 34052527423): asking does nothing; lied mid-answers rewrite ~60% of
second presses. Methods:
[ask-mid-traj-qwen38-n16-v1-methods.md](ask-mid-traj-qwen38-n16-v1-methods.md);
story: [ask-mid-trajectory-story.md](ask-mid-trajectory-story.md). CPU analysis:
`analyze_ask_mid_trajectory.py` (immutable). Claim rows C19–C21 are ledger
indices, not alternate experiment names.

## Rename-invariant tool/button check (canonical ID: `rename_invariant`)

Protocol:
[rename-invariant-protocol.md](rename-invariant-protocol.md) +
[protocol_rename_invariant.json](protocol_rename_invariant.json).

Canonical protocol / task / study ID: **`rename_invariant`**. Prepare:
`--task rename_invariant`. Neutral aliases; crossmap names and consequences
(both directions). **CPU dry-run only** — Actions refuses `dry_run=false` until
an analyzer + documented results layout exist. Module:
`src/jspace_policy/rename_invariant.py`.

## Ask-after-the-act (design stub)

Stub only — not wired, not scored:
[ask-after-the-act-protocol.md](ask-after-the-act-protocol.md). Operator
priority ordering lives in the decision log, not here.

## Prepared-payload permanence

Scored runs write `results/report_reactivity/<run_id>/prepared.json` (exact
CPU prepare dict; sha256 must match `raw.json`’s `payload_sha256`). Local
scratch `artifacts/prepared/*` stays gitignored. GPU Actions uploads now also
include the prepare file as a short-lived backup. Paid runs require
`pinned_payload_sha256` to match the freshly prepared hash before Modal.

## Next protocol draft (not executed)

[branched_preaction_auditor_protocol.md](branched_preaction_auditor_protocol.md)
and stub [protocol_bpa.json](protocol_bpa.json): shared-prefix fork into (A) tool
choice vs (B) pre-action predictions of verifiable/unverifiable auditor reports.
Primary estimand is pre-action honesty-intent relative to C5's post-action null.
Mapping-conditional reporting is mandatory (C10).
