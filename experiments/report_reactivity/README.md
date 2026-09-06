# Report-reactivity experiments

CPU prepare lives here (`prepare.py`, `prepare_preflight.py`, `protocol.json`).
GPU scoring is the root Modal entrypoint `modal_report_reactivity.py`.

Dispatch via `.github/workflows/report-reactivity.yml` (`workflow_dispatch`).
`dry_run` defaults to true so Actions can validate and materialize payloads
without Modal GPU spend; set it false only for an intentional scoring run.
Default `batch_size` is 4 (see claim ledger C1).
