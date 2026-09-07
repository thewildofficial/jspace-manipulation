# Current research record

Status: **reconciled record for the report-reactivity merge sequence**. This
page records the state of GitHub `main` at commit
[`30fa352`](https://github.com/thewildofficial/when-words-override-consequences/commit/30fa3520d954d8875ec84a4f3c23739fbe220664)
after merges #18–#24 and the gate-failure archive. It is a navigation and
status record, not a new result.

## Status vocabulary

Use exactly one of these states for every experiment or run:

- **Design only:** protocol and CPU code exist; no payload has passed prepare.
- **Prepared:** CPU payload passed tests and was written, but no GPU score ran.
- **Attempted — gate failed:** a run started, but an instrument or parity gate
  withheld scores. This is not behavioral data.
- **Discovery scored:** scores passed the instrument gate, but the result is a
  pilot or descriptive finding rather than a locked confirmation.
- **Locked confirmation:** a frozen contrast was scored on held-out data.

Never call a design, a dry-run, or a gate failure an “experiment result.”

## Identity rules

These fields have different jobs and must not be substituted for one another:

| Field | Meaning | Example |
|---|---|---|
| `track` | Research area | `report-reactivity` |
| `study_id` | Same canonical ID as the protocol/task; retained for payload compatibility | `ask_mid_trajectory` |
| `protocol_id` | Code/CLI enum | `ask_mid_trajectory` |
| `run_id` | One concrete execution | `ask-mid-traj-qwen38-n16-v1` |
| `claim_id` | What a scored run supports | `C19`, `C20`, `C21` |
| `queue_id` | Scheduling only | `P7` |

Existing artifact names stay unchanged for reproducibility. For live
report-reactivity work, the protocol ID is the sole scientific identity and is
copied into `study_id`, `task`, and payload `protocol_id`. In new prose, use
that canonical ID and then give the run ID once. Queue labels such as `#6`,
`#7`, and “PriGo” are operational metadata, not experiment names. Do not
introduce another V/RBG/E-number layer for this track. The detailed rule is
[`experiments/report_reactivity/NAMING.md`](../../experiments/report_reactivity/NAMING.md).

## Recent merge sequence

| Merge | What changed | Scientific standing |
|---|---|---|
| #15–#17 | Actions workflow, CPU dry-run, spend ledger, Modal JSON return | Infrastructure; no behavioral result |
| #18 | 16-base ask-first baseline and swapped-report analysis | Discovery baseline; Direct was at ceiling |
| #19 | Harder-game attempt and poisoned-self-talk cell map | Discovery; difficulty manipulation did not break the ceiling |
| #20 | Harder-games result plus mid-trajectory protocol | Discovery null plus a new protocol; no locked claim |
| #21 | Permanent prepared inputs and immutable analysis outputs | Archival repair; no new behavioral result |
| #22 | Mid-trajectory scoring and rename-invariant CPU wiring | C19–C21 are discovery-only; rename-invariant was not yet a result |
| #23 | Fixed a nonce collision that spelled `SAVE` | Engineering repair; the subsequent GPU attempt failed parity |
| #24 | Unified live IDs and added fail-closed rename/pinned-payload gates | Process repair; future rename GPU dispatch is refused until unlocked |

## Current report-reactivity status

### Mid-trajectory ask — `ask_mid_trajectory`

Run: `ask-mid-traj-qwen38-n16-v1` (`ask_mid_trajectory`). The scored discovery
pilot supports:

- C19: self mid-ask versus matched no-ask control produced a zero intervention
  contrast in this scaffold.
- C20: an injected inverted mid-answer changed the second press on 59.375% of
  the 16-base pilot trials.
- C21: the opaque/prose split is descriptive only.

These are not locked confirmations, mechanism claims, HITL-general claims, or
evidence of scheming. The durable raw, prepared, manifest, and analysis files
are documented by PR #22 and should remain the source of truth.

### Post-controls v1 — working tree only

The local `post-controls-v1` material is an exploratory follow-up, not part of
the merged-main record and not yet assigned a claim ID. Its raw output and
manifest exist, but the exact prepared payload remains under gitignored
`artifacts/processed/post_controls_v1.json`. See [`docs/post/README.md`](../post/README.md)
for the archive condition and scope limits.

### Rename-invariant tool/button check — `rename_invariant`

Protocol/task/study ID: `rename_invariant`. `RENAME-INVARIANT-1` is a
historical alias only. The experiment currently has no behavioral result.

The engineering trail is:

1. [Run 34053946547](https://github.com/thewildofficial/when-words-override-consequences/actions/runs/34053946547)
   failed during CPU prepare because `DSAVEKA004Q` contained `SAVE`.
2. [Run 34054166742](https://github.com/thewildofficial/when-words-override-consequences/actions/runs/34054166742)
   passed the corrected dry-run: 150 tests passed, 4 skipped; 288 queries and
   128 records prepared.
3. [Run 34054251019](https://github.com/thewildofficial/when-words-override-consequences/actions/runs/34054251019)
   was a GPU attempt, despite PR #23's “No GPU” note. Batch/singleton parity
   was `0.375` against a `0.25` threshold, so the instrument gate failed and
   `scores` was empty. The reserved ceiling was approximately `$0.783`.

The durable engineering record is
[`results/report_reactivity/rename-invariant-qwen38-n16-v1/`](../../results/report_reactivity/rename-invariant-qwen38-n16-v1/).
Do not cite this run as rename-following, consequence-following, or any other
behavioral finding.

## Remaining record work

- Keep the failed rename attempt visible as a gate event, not as a claim.
- Do not dispatch `rename_invariant` with `dry_run=false` until its scoring
  analyzer and permanent result path exist, or explicitly label any attempt as
  an engineering gate run.
- PR #24 now enforces the rename GPU lock and requires a reviewed
  `pinned_payload_sha256` before paid scoring.
- Keep the uncommitted `post-controls-v1` work separate until it is reviewed;
  it is not part of the merged-main record represented here.
