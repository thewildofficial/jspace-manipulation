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

## 2026-09-06 — RR-D002 — Record C14 first clean GHA GPU preflight (no new science)

**Context.** After PR #17 (`score_gpu` JSON-string return), human
`workflow_dispatch` `gha-preflight-38-v4` (Actions run
[34046322701](https://github.com/thewildofficial/when-words-override-consequences/actions/runs/34046322701),
`dry_run=false`, stage=`preflight`, `batch_size=4`) completed all steps:
CPU prepare (payload sha256 `226e488f…`, `n_queries=2`), GHA-ledger reserve,
Modal A100 scoring, local `raw.json` write, artifact upload. Artifact parity:
`passed=true`, `batch_single_max_abs=0.125`, `choices_agree=true`,
`replay_max_abs=0.0`. Spend: second `reservations_gha.jsonl` row
(`ceiling≈0.7829`, cumulative `total_reserved_usd≈1.5657` / $28). Historical
`reservations.jsonl` unchanged.

**Decision.** Commit immutable
`results/report_reactivity/gha-preflight-38-v4/{raw.json,input_manifest.json}`
and the updated GHA ledger; ledger as **C14** engineering confirmation only.

**Not claimed.** Behavioral/science estimands; baseline/incident/replication
path validity; paper conclusions. Preflight tokenizer smoke only.
