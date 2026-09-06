# Methods note: GitHub Actions report-reactivity path (C11 / C12 / C13)

Status: **engineering / ops findings**, not behavioral results.
Companion ledger rows: **C11** (CPU dry-run success), **C12** (GPU reserve
failure before scoring), **C13** (GPU scored remotely then local torch
deserialize failure). Decision: [RR-D001](../../docs/next-sprint/decision-log.md).

## C11 — CPU dry-run confirmed

| Field | Value |
|---|---|
| Actions run | [34044659196](https://github.com/thewildofficial/when-words-override-consequences/actions/runs/34044659196) |
| `run_id` | `gha-dryrun-preflight-38-v1` |
| `dry_run` | `true` (no Modal / no GPU) |
| Stage | `preflight` |
| Model | `Qwen/Qwen3.8-27B` |
| Revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| CPU gate | `uv run --extra dev pytest -q` → 128 passed, 4 skipped |
| Prepare | `experiments/report_reactivity/prepare_preflight.py` |
| Prepared payload sha256 | `226e488f85f437b14ff4a66382e73c4f36a8d7be8b3dde315fc98c9e8105fb20` |
| `n_queries` | 2 (prompt lengths 24 and 28 tokens) |
| Artifact | uploaded prepared JSON; Modal step skipped |

Pipeline: input validation → pytest → tokenizer prepare → artifact upload →
dry-run exit. Not a behavioral claim.

## C12 — GPU dispatch failed at ledger `reserve()` (no scores)

| Field | Value |
|---|---|
| Actions run | [34044902792](https://github.com/thewildofficial/when-words-override-consequences/actions/runs/34044902792) |
| `run_id` | `gha-preflight-38-v1` |
| `dry_run` | `false` |
| CPU prepare | succeeded; same payload sha256 `226e488f…` |
| Modal credentials | present (`Refuse missing Modal credentials` passed) |
| Modal app | initialized (`Created function score_gpu`) |
| `input_manifest.json` | written under `results/report_reactivity/gha-preflight-38-v1/` with `ceiling_usd≈0.782856`, `payload_sha256=226e488f…`, `stage=preflight` (run_id is the results directory name / workflow input; pre-fix manifests did not yet embed `run_id`) |
| Failure | `sprint_runtime.reserve` → `ValueError('global or stage budget exhausted')` |
| GPU scoring | **did not run** (no `raw.json`) |
| Reservation row | **not** appended to historical `reservations.jsonl` |
| Upload step | still ran (`if: always() && !dry_run`); shipped manifest dir + reservations copy |

### Why reserve failed (ledger / stage-cap, not missing money)

Historical `results/report_reactivity/reservations.jsonl` total ≈ **$7.26** of
the $30 study ceiling. Stage `preflight` alone sums to ≈ **$1.5657** against
`STAGE_LIMITS['preflight']=2.0`. Another ceiling of ≈ **$0.7829** cannot fit
(`1.5657 + 0.7829 > 2.0`). Provider Modal account balance (~$28) and valid
tokens are irrelevant to this stop: fail-closed study accounting blocked
dispatch before any GPU forward pass.

## C13 — Ledger OK; remote score returned; local torch deserialize failed

| Field | Value |
|---|---|
| Actions run | [34045326136](https://github.com/thewildofficial/when-words-override-consequences/actions/runs/34045326136) |
| `run_id` | `gha-preflight-38-v2` |
| `dry_run` | `false` |
| Stage | `preflight` |
| CPU prepare | succeeded; payload sha256 `226e488f…` |
| Ledger | `REPORT_REACTIVITY_LEDGER=results/report_reactivity/reservations_gha.jsonl` |
| Reserve | **succeeded** — row retained (`ceiling≈0.7829`, `global_ceiling_usd=28`, `total_reserved_usd≈0.7829`) |
| `input_manifest.json` | written (`ledger_path` set, `global_ceiling_usd=28`) |
| Remote | Modal loaded weights and returned from `score_gpu.remote` |
| Failure | Local client: `DeserializationError` / `ModuleNotFoundError: No module named 'torch'` |
| `failure.json` | `{"automatic_retry": false, "error": "Deserialization failed because the 'torch' module is not available in the local environment.", "reservation_retained": true}` |
| Scores | **no** usable `raw.json` for analysis |
| Historical ledger | unchanged (correct) |

### Why deserialize failed (payload pickle, not missing GPU)

`uv run --extra modal` on the Actions CPU runner installs Modal but not torch.
Modal's default return path pickles the remote result; anything that still
requires the `torch` module on unpickle fails on the client even after the GPU
job finishes. Installing full torch on the runner is rejected; keep the return
payload JSON-safe instead.

### Fix — JSON string return (no torch on Actions)

| Choice | Detail |
|---|---|
| Sanitize | `sprint_runtime.to_jsonable` converts tensors/arrays to plain Python |
| Remote return | `score_gpu` → `dumps_jsonable(result)` (JSON `str`) |
| Local parse | `loads_jsonable(...)` before `write_new(.../raw.json)` |
| Regression | `test_score_return_payload_is_json_serializable_without_torch` |
| Not done | Adding torch to the Actions CPU environment |

**Still not claimed:** any behavioral score for `gha-preflight-38-v2`.
Re-dispatch with a **new** `run_id` after this fix merges.

## Fix implemented (RR-D001) — new GHA ledger, historical immutable

| Choice | Detail |
|---|---|
| New ledger path | `results/report_reactivity/reservations_gha.jsonl` |
| Global ceiling | **$28.0** (aligned to reported Modal balance) |
| Stage limits | unchanged `STAGE_LIMITS` (counters restart on the new empty file) |
| Env var | `REPORT_REACTIVITY_LEDGER` — default historical path for local continuity |
| Workflow | sets `REPORT_REACTIVITY_LEDGER=results/report_reactivity/reservations_gha.jsonl` |
| Optional | `REPORT_REACTIVITY_GLOBAL_CEILING` override |

Rejected: mutating historical rows or silently raising the old $30 ledger's
preflight stage cap in place.

**Still not claimed:** any GPU behavioral score for `gha-preflight-38-v1`.
Re-dispatch with a **new** `run_id` after this wiring merges (same run_id would
also be refused once reserved).

## Reproduction (CPU only)

```bash
uv sync --extra dev
uv run --extra dev pytest -q
uv run --with 'transformers>=5.5' --with 'huggingface-hub>=0.34' \
  --with 'httpx[socks]>=0.28' --with 'jinja2>=3.1' \
  python experiments/report_reactivity/prepare_preflight.py \
  --model Qwen/Qwen3.8-27B \
  --output /tmp/gha-dryrun-preflight-38-v1.json
# Expect sha256 226e488f85f437b14ff4a66382e73c4f36a8d7be8b3dde315fc98c9e8105fb20
```

No Modal/GPU from documentation agents.
