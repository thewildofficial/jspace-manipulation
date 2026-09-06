# Report-reactivity experiments

CPU prepare lives here (`prepare.py`, `prepare_preflight.py`, `protocol.json`).
GPU scoring is the root Modal entrypoint `modal_report_reactivity.py`.

Dispatch via `.github/workflows/report-reactivity.yml` (`workflow_dispatch`).
`dry_run` defaults to true so Actions can validate and materialize payloads
without Modal GPU spend; set it false only for an intentional scoring run.
Default `batch_size` is 4 (see claim ledger C1).

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

## Next protocol draft (not executed)

[branched_preaction_auditor_protocol.md](branched_preaction_auditor_protocol.md)
and stub [protocol_bpa.json](protocol_bpa.json): shared-prefix fork into (A) tool
choice vs (B) pre-action predictions of verifiable/unverifiable auditor reports.
Primary estimand is pre-action honesty-intent relative to C5's post-action null.
Mapping-conditional reporting is mandatory (C10).
