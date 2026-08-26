# Stage 1E.1 — Protocol V2 workspace mapping

## The question

Before looking at the flexible test outcomes, which layers should count as the
model’s middle “workspace” for the V2 intervention?

This was a neutral, task-independent measurement. It used unrelated prompts
and a fixed rule based on two simple signals: how unusual the lens values were
and how closely they agreed with the model’s ordinary output readout.

## What happened

The rule selected **layers 36–43**, or normalized depth 0.571–0.683. The band
was selected before the flexible-generalization outcomes were opened.

This is a reproducible selection under one operational rule. It does not mean
that every task uses the same layers. In fact, the later H0 failure showed why a
task-independent band should not automatically be treated as a validated
causal workspace.

## Evidence files

- [`raw/workspace_diagnostics.jsonl`](raw/workspace_diagnostics.jsonl): per-layer, per-position measurements.
- [`workspace_band.json`](workspace_band.json): machine-readable selected band.
- [`summaries/workspace_layer_summary.csv`](summaries/workspace_layer_summary.csv): deterministic layer summary.
- [`figures/workspace_selection.png`](figures/workspace_selection.png): the selection plot.
- [`run_manifest.json`](run_manifest.json): model, lens, config, and run metadata.

The selection rule is executed from [`../../modal_v2.py`](../../modal_v2.py) and
the committed figure is rebuilt by [`../../scripts/analyze_v2_h0.py`](../../scripts/analyze_v2_h0.py).

The original H0 result that used this band is in [`../v2_smoke_tests/README.md`](../v2_smoke_tests/README.md).

[Results map](../README.md) · [V2 documentation](../../docs/v2/README.md)
