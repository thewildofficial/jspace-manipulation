# Methods note: GitHub Actions CPU dry-run path (C11)

Status: **engineering confirmation**, not a behavioral result.
Companion ledger rows: **C11** (confirmed), **C12** (placeholder; no GPU claim).

## What was confirmed

Workflow `.github/workflows/report-reactivity.yml` (`workflow_dispatch` only)
was merged on main (PR #15). A CPU-only dispatch succeeded:

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

Pipeline exercised: input validation → pytest → tokenizer prepare → artifact
upload → dry-run exit. Default `batch_size=4` (C1 parity lesson) is wired for
future GPU dispatches but was unused here.

## What is not claimed

- No model action accuracy, violation rate, auditor truthfulness, or parity score.
- Preflight queries are instrument checks ("return only A or B"), not incident-desk
  episodes.
- Success of CPU prepare does not imply Modal credentials, image build, reservation
  headroom, or GPU scoring will succeed.

## GPU preflight placeholder (C12; not claimed)

A subsequent dispatch with `run_id=gha-preflight-38-v1` and `dry_run=false`
(Actions run [34044902792](https://github.com/thewildofficial/when-words-override-consequences/actions/runs/34044902792))
passed CPU prepare (same prepared sha256) and credential presence checks, then
failed locally at `sprint_runtime.reserve` with
`ValueError: global or stage budget exhausted` before any GPU forward pass.
Study ledger already holds ~$7.26 reserved historically; preflight stage ceiling
in `STAGE_LIMITS` is $2.0 and was already consumed by earlier local preflights.
No behavioral GPU artifact exists for this run_id. Do not fill C12 until a
successful scored artifact is committed.

Provider account balance (~$28 remaining on Modal at last user report) is
orthogonal to the fail-closed study ledger: provider headroom does not authorize
exceeding `STAGE_LIMITS` / the $30 global ceiling.

## Reproduction (CPU only)

```bash
# Local analogue of the Actions dry-run prepare step:
uv sync --extra dev
uv run --extra dev pytest -q
uv run --with 'transformers>=5.5' --with 'huggingface-hub>=0.34' \
  --with 'httpx[socks]>=0.28' --with 'jinja2>=3.1' \
  python experiments/report_reactivity/prepare_preflight.py \
  --model Qwen/Qwen3.8-27B \
  --output /tmp/gha-dryrun-preflight-38-v1.json
# Expect sha256 226e488f85f437b14ff4a66382e73c4f36a8d7be8b3dde315fc98c9e8105fb20
```

Or re-dispatch Actions with `dry_run=true` and a fresh `run_id`. Never set
`dry_run=false` from an agent session without explicit human budget approval and
ledger headroom.
