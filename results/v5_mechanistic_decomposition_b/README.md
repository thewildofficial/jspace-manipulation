# V5-RBG-5B results

This directory is reserved for the immutable RBG-5B run. The checked-in
configuration and dataset manifest are frozen; behavior, discovery, locked
patching, J-space, and final analysis artifacts are created append-only by the
`modal_v5b_mechanistic_decomposition.py` entrypoints.

The one-time dataset materialization command is retained for auditability but
must not be run against this already frozen checkout (the entrypoint refuses to
overwrite `dataset.json`):

```text
modal run modal_v5b_mechanistic_decomposition.py::app.rbg5b_freeze_dataset
```

Normal execution order:

```text
modal run modal_v5b_mechanistic_decomposition.py::app.rbg5b_preflight
modal run modal_v5b_mechanistic_decomposition.py::app.rbg5b_behavior
modal run modal_v5b_mechanistic_decomposition.py::app.rbg5b_discovery
modal run modal_v5b_mechanistic_decomposition.py::app.rbg5b_locked
modal run modal_v5b_mechanistic_decomposition.py::app.rbg5b_jspace
uv run python scripts/analyze_v5_mechanistic_decomposition_b.py
```

No result is implied until the corresponding immutable manifest exists.

## GitHub Actions fallback

Local `modal run` needs a working Modal control-plane connection.  If the local
client reports `Could not connect to the Modal server`, push the frozen commit
and dispatch [`.github/workflows/v5-mechanistic-decomposition-b.yml`](../../.github/workflows/v5-mechanistic-decomposition-b.yml)
with `stage=all`.  The GitHub runner only orchestrates the same Modal functions;
the A100 work and the pinned model still run in Modal.  The repository must
have `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` Actions secrets.  The workflow
uploads the local manifests, logs, downloaded Modal-volume artifacts, and
offline analysis as an immutable run artifact.

From the repository checkout, the manual dispatch is:

```text
gh workflow run v5-mechanistic-decomposition-b.yml --ref <frozen-branch> -f stage=all
gh run watch
```

Stage-specific dispatches are intended for preflight/diagnostics or a checkout
that already contains the predecessor manifests.  A fresh end-to-end run
should use `stage=all`, because GitHub workspaces are ephemeral between runs.
