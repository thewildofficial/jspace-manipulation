# Post-controls v1

Status: **exploratory working-tree record; not part of merged `main`**.

This is a 16-base Qwen3.8-27B follow-up comparing false-report position and
source framing in a constrained A/B choice interface and a constrained A/B
simulator-dispatch interface. It has no mechanistic or confirmatory claim.

## Evidence and reproducibility

- [Protocol](controls-protocol.md): frozen design and scope limits.
- [Draft readout](draft.md): corrected narrative; `draft-v1.md` is the earlier
  review draft.
- [Cell table](controls-table.md) and [summary](controls-summary.json):
  per-cell counts.
- [Execution traces](execution-traces.json): scored choices and simulator
  outcomes.
- [Raw GPU output](../../results/report_reactivity/post-controls-v1/raw.json)
  and [manifest](../../results/report_reactivity/post-controls-v1/input_manifest.json).
- Preparation and analysis: `experiments/report_reactivity/post_controls.py`
  and `scripts/analyze_post_controls.py`.

The exact prepared payload is currently
`artifacts/processed/post_controls_v1.json`, which is gitignored and therefore
not yet reproducible from a fresh checkout. Before this is merged or cited as
a durable result, preserve that exact payload (or a byte-identical copy), its
hash, and the code/protocol hashes alongside the raw output.

## Interpretation

The corrected historical result is 22 false-report errors out of 128 Qwen3.8
variants (17.2%), not 106 false-report-following actions. The follow-up reports
large before/after differences in several small cells, but the source framing
and constrained tool-role conditions are not clean internal-role interventions.
Use this as a descriptive pilot only, and assign a claim ID only after the
prepared payload is archived and the record is reviewed.
