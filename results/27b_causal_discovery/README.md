# V1-CD-1 — causal discovery: readable, not writable

## The question

The observational pilot found an internal pattern that changed when the rule
changed. Could we now write that pattern back into the model and make it report
the other answer, while checking that the underlying fact was still present?

## What happened

The answer was no for this particular candidate. At the fixed layer-47 test
point, the intervention changed a score by **0.091 log-odds**, but produced
**0/48 public report flips**. A separate fact proxy moved by **4.11 SD**, and the
residual state moved by **15.7% RMS**. The policy ablation and the reveal↔conceal
coordinate swap were nearly inert.

In plain language: the candidate was visible to the lens, but the write did not
behave like a reliable “change the reporting rule” control. This is a negative
causal result about the tested one-dimensional object. It does not show that no
reporting policy exists anywhere in J-space.

Because the candidate never became the frozen policy instrument, transfer and
strategic-monitoring stages were not opened. The later V2–V5 follow-ons are
separate protocol generations and are documented separately.

See [`docs/causal-discovery-report.md`](../../docs/causal-discovery-report.md) for the complete design, figures, interpretation, limitations, hypothesis decisions, and cost accounting.

## Files

- `raw/interventions_discovery.parquet`: five-layer, four-dose discovery sweep.
- `raw/interventions_discovery.manifest.json`: exact run identity and intervention registry.
- `raw/interventions_stability.parquet`: fixed-point six-scenario stability check.
- `raw/interventions_stability.manifest.json`: exact stability-run identity.
- `summaries/`: 5,000-resample scenario-clustered summaries.
- `figures/`: publication-oriented causal and behavioral figures.
