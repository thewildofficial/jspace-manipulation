# Report-reactivity experiments

CPU prepare lives here (`prepare.py`, `prepare_preflight.py`, `protocol.json`).
GPU scoring is the root Modal entrypoint `modal_report_reactivity.py`.

Dispatch via `.github/workflows/report-reactivity.yml` (`workflow_dispatch`).
`dry_run` defaults to true so Actions can validate and materialize payloads
without Modal GPU spend; set it false only for an intentional scoring run.
Default `batch_size` is 4 (see claim ledger C1).

## Instrumentation confirmation (C11)

See [gha-cpu-dryrun-methods.md](gha-cpu-dryrun-methods.md): Actions CPU dry-run
`gha-dryrun-preflight-38-v1` confirmed pytest + prepare + artifact upload.
GPU preflight via Actions is **not** claimed (ledger placeholder C12).

## Next protocol draft (not executed)

[branched_preaction_auditor_protocol.md](branched_preaction_auditor_protocol.md)
and stub [protocol_bpa.json](protocol_bpa.json): shared-prefix fork into (A) tool
choice vs (B) pre-action predictions of verifiable/unverifiable auditor reports.
Primary estimand is pre-action honesty-intent relative to C5's post-action null.
Mapping-conditional reporting is mandatory (C10).
