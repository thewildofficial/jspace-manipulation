# Report-reactivity decision log

Living operational decisions for the report-reactivity / incident-desk track.
Scientific claims remain in [claim-ledger.md](claim-ledger.md).

## 2026-09-06 — RR-D001 — Start a GHA-era reservation ledger (do not mutate historical)

**Context.** Actions GPU dispatch `gha-preflight-38-v1` (run 34044902792) passed
CPU prepare and Modal credential/app init, wrote `input_manifest.json`
(`ceiling_usd≈0.7829`, `payload_sha256=226e488f…`, `stage=preflight`), then
failed at `sprint_runtime.reserve` with
`ValueError('global or stage budget exhausted')` before any GPU forward pass.
Historical `results/report_reactivity/reservations.jsonl` already holds
~$7.26 total; stage `preflight` alone holds ~$1.5657 against
`STAGE_LIMITS['preflight']=2.0`, so another ~$0.7829 ceiling cannot fit.
Provider Modal balance (~$28) was not the blocker — this is a **study ledger /
stage-cap** fail-closed stop. Upload-artifacts still ran (`always()`), shipping
the manifest directory plus a reservations copy.

**Decision.** Keep historical `reservations.jsonl` **immutable**. Open a fresh
GHA-era ledger at `results/report_reactivity/reservations_gha.jsonl` with
**global ceiling $28.0** (aligned to the reported Modal balance), same
`STAGE_LIMITS` map (stage counters restart at zero on the new file). Wire
selection through env var `REPORT_REACTIVITY_LEDGER`, defaulting to the
historical path for local continuity; the GitHub workflow sets it to the GHA
ledger. Optional override: `REPORT_REACTIVITY_GLOBAL_CEILING`.

**Rejected alternative.** Raising `STAGE_LIMITS['preflight']` or the historical
global $30 ceiling in-place would mix GHA spend into a ledger whose rows already
encode the completed $30-study allocation and fail-closed retentions. Prefer
separate files over rewriting history.

**Not claimed.** No GPU scores; no behavioral result; no authorization to exceed
provider balance. First GHA GPU success still requires a human `dry_run=false`
dispatch after this wiring merges.

## 2026-09-06 — RR-D002 — Harder-games null; next bet = mid-trajectory ask

**Context.** Harder-games Actions run 34051102729 (`harder-games-qwen38-n16-v1`,
`history_mode=redundant`) completed with parity pass. Direct and all
non-swapped arms remained at ceiling 1.0; swapped ≈0.789 with worst cell
prose/opposed/strategic = 0.5. Ask-first rescue stays unidentifiable on this
scaffold (C17/C18).

**Decision.** Record the substantive null in the claim ledger and story notes.
Do **not** spend further GPU on redundant-demo dials hoping to unlock ask-first
rescue without a new content-controlled difficulty idea. Next executable bet:
**mid-trajectory ask-as-intervention** (`ask_mid_trajectory`) — consequence
question between two forced presses vs matched non-consequence chatter. Wire
CPU prepare + Actions `task=ask_mid_trajectory` dry-run; score only after dry
artifact hash is pinned. No Modal from the docs/code PR.

**Rejected alternative.** Opening a second instrument-only PR for mid-trajectory
plumbing without the harder-games scientific null.

**Not claimed.** That demos make games harder in a useful way; locked
confirmation; any mid-trajectory behavioral result (not yet scored).

## 2026-09-06 — RR-D003 — Mid-trajectory scored; PriGo priority order

**Context.** Mid-trajectory Actions run 34052527423 (`ask-mid-traj-qwen38-n16-v1`,
payload sha256 `50b854ae…`, parity `batch_single_max_abs=0.125`) completed.
Asking does nothing (self−control contrast 0.0 on flip and choice2); a lied
mid-answer rewrites ~60% of second presses (swapped flip 0.59375). Claim rows
C19–C21. Harder-games ask-first rescue remains blocked (C17); mid-ask is not the
intervention lever on this scaffold.

**Decision — priority order (PriGo):**

1. **#7 mid-trajectory** — **done** (scored discovery + claims; no further GPU
   on ask-as-intervention without a new design).
2. **#6 rename-invariant** — **next** (MOST NOVEL): when only names/labels
   change, does the second press follow labels or outcomes? Wire CPU prepare +
   Actions `task=rename_invariant` dry-run; neutral aliases; crossmap both
   directions. Score only after dry-run hash is pinned. No Modal from the
   docs/wire PR.
3. **#4 ask-after-the-act** — protocol stub only for now (queue visibility);
   do not block #6.
4. **#1 lie titration** — after #4 design is ready to wire.
5. **#2 attribution** — after #1.

**Rejected alternative.** Spending further GPU to “confirm” mid-ask intervention
on the same scaffold after a clean zero contrast; or jumping to attribution
before the rename-invariant behavioral screen.

**Not claimed.** Rename-invariant behavioral results (not scored); ask-after
effects; scheming; locked confirmation of C19–C21.
