# V5-RBG-5B results

**Second-pass correction:** the archived candidate J-lens path omitted the final
model normalization, while full-vocabulary rows used it. Its zero-state parity
check could not detect the difference. Readout percentages below are historical
outputs awaiting corrected computation, not evidence of near-chance J-lens
performance. The failed identity control also limits mechanistic interpretation.
The frozen causal endpoint remains unsupported; no archived outputs are changed.
See [audit](../../docs/next-sprint/audit.md).

The following paragraphs preserve the sequence of execution and recovery;
earlier descriptions of unopened stages are historical, not current status.

This directory contains the immutable surviving artifacts from GitHub Actions
run [`33337232212`](https://github.com/thewildofficial/when-words-override-consequences/actions/runs/33337232212).
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
[`33377569076`](https://github.com/thewildofficial/when-words-override-consequences/actions/runs/33377569076)
completed in 922.54 seconds. It froze opaque-donor residual replacement at layer
40 and `payoff_end`: cluster-mean correct-action margin change `+0.05882`, exact
sign-flip `p = 0.02145`, 34 bases and 60 rows. This was a small directional
effect: one row reached a counted repair through a zero-margin tie, while no row
became strictly positive-margin correct. The untouched locked split is now
legally open under the prospective positive-winner rule; no locked or J-space
result is yet implied.

Locked-only run
[`33380084999`](https://github.com/thewildofficial/when-words-override-consequences/actions/runs/33380084999)
then reached its exact 3,000-second Modal timeout while performing serial
descriptive probe bootstraps, before the frozen patch or any locked manifest was
written. This is infrastructure censoring, not a negative causal result: the
locked endpoint remains unopened and J-space was correctly skipped. Decision
V5-D018 preserves every frozen scientific choice but moves the algebraically
equivalent probe bootstrap to the GitHub runner CPU. The authorized salvage has
a combined locked-plus-J-space Modal ceiling of USD 1.93225 against USD 2.50
remaining, with retries disabled.

The recovered locked result is now complete. GPU capture and the frozen natural
interchange ran in 319.23 seconds on the A100; geometry, probe predictions, and
all 2,000-resample statistics ran locally on CPU in 44.26 seconds. The primary
endpoint repaired **2/50 rows (4%)**, with mean correct-action margin change
`+0.020` but exact base-cluster sign-flip `p = 0.42924`; this fails the required
20% repair, positive-margin, and `p<.05` conjunction. Consequence reports stayed
at 100% before and after. Non-damage controls stayed at 100%, while the strict
identity-margin control failed (maximum absolute shift `0.5`), so the overall
selective-causal-transport claim is **not supported**. This result does not
establish that the semantic-action effect is a robust causal bottleneck.

The workflow badge is red because its first local post-processing attempt lacked
the already-frozen discovery probe file; the GPU artifact and all hashes were
preserved. The small manifest and derived CPU files were reconstructed from the
immutable [run 33391981859](https://github.com/thewildofficial/when-words-override-consequences/actions/runs/33391981859)
artifact without rerunning the model. J-space remains a separately gated,
observational readout and has not been used to reinterpret this negative result.

The required J-space secondary run
[`33395448789`](https://github.com/thewildofficial/when-words-override-consequences/actions/runs/33395448789)
completed in 509.305 seconds and wrote 69,120 rows (34,560 per split), SHA
`6b17bca2d1cbe822cb4dbf654557ec1a1b8e49c9c8568da796004b712f7e1cbe`. Projected
action-token top-1 ranking was 50.96% on discovery and 54.69% on locked rows;
within locked rows it was 57.73% for opposed games and 42.56% for aligned games.
These are lens/readout descriptives, not evidence that J-space controls the
model, and they cannot select a new patch site or overturn the failed locked
endpoint. The J-space measured Modal subtotal was USD 0.47919 (buffered ledger
amount USD 0.57503); the two successful salvage stages together used USD 0.81301
measured / USD 0.97561 buffered.

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

Local `modal run` needs a working Modal control-plane connection. If the local
client reports `Could not connect to the Modal server`, push the frozen commit
and dispatch [`.github/workflows/v5-mechanistic-decomposition-b.yml`](../../.github/workflows/v5-mechanistic-decomposition-b.yml)
with the next unopened stage. The GitHub runner orchestrates the Modal GPU
functions and performs CPU-only analysis locally; the A100 work and pinned model
still run in Modal. The repository must
have `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` Actions secrets.  The workflow
uploads the local manifests, logs, downloaded Modal-volume artifacts, and
offline analysis as an immutable run artifact.

From the repository checkout, the manual dispatch is:

```text
gh workflow run v5-mechanistic-decomposition-b.yml --ref <frozen-branch> -f stage=locked
gh run watch
```

The checked-in behavior artifact now permits a stage-specific `discovery`
dispatch without repeating the passed behavior inference. Later stages are
dispatched only after their predecessor manifests have been reviewed and
committed.
