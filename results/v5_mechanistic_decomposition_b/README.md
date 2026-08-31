# V5-RBG-5B results

This directory contains the immutable surviving artifacts from GitHub Actions
run [`33337232212`](https://github.com/thewildofficial/jspace-manipulation/actions/runs/33337232212).
The corrected preflight and behavioral gate passed. Discovery then timed out
inside descriptive probe fitting before its manifest or Modal-volume artifact
was committed; locked patching and J-space were never opened.

The passing behavioral result is not rerun: option reports were 98.10% correct,
aligned and opaque action accuracy were both 100%, and redundant semantic
history reduced opposed prose-assertion action accuracy by 33.85 percentage
points (exact base-cluster `p = 2.997e-10`). The complete frozen gate is in
[`analysis/behavior_analysis.json`](analysis/behavior_analysis.json).

Run `33337232212` emitted 670 liblinear convergence warnings before discovery
hit its 3,000-second timeout. Decision V5-D016 records the prospective
implementation-only repair and the new $6.60 remaining authorization.

The repaired discovery-only run
[`33377569076`](https://github.com/thewildofficial/jspace-manipulation/actions/runs/33377569076)
completed in 922.54 seconds. It froze opaque-donor residual replacement at layer
40 and `payoff_end`: cluster-mean correct-action margin change `+0.05882`, exact
sign-flip `p = 0.02145`, 34 bases and 60 rows. This was a small directional
effect: one row reached a counted repair through a zero-margin tie, while no row
became strictly positive-margin correct. The untouched locked split is now
legally open under the prospective positive-winner rule; no locked or J-space
result is yet implied.

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

The checked-in behavior artifact now permits a stage-specific `discovery`
dispatch without repeating the passed behavior inference. Later stages are
dispatched only after their predecessor manifests have been reviewed and
committed.
