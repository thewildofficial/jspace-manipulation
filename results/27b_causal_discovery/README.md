# Qwen3.6-27B Stage-1 causal discovery

The middle-layer reporting-policy family is observationally readable but failed the preregistered causal gate.

At the fixed layer-47, 16-SD stability point across six base scenarios, the world-conditioned effect was **0.091 log-odds** (scenario-bootstrap 95% interval **[0.086, 0.102]**), with **0/48 public report flips**. The provisional fact proxy moved **4.11 SD**, the final-position residual moved **15.7% RMS**, and the matched private fact-use margin decreased by **0.125 log-odds**. Policy ablation and reveal↔conceal coordinate swap were nearly inert.

This is a negative causal result, not evidence that reporting policies never exist in J-space. The tested one-dimensional seed-family object did not become `frozen_policy_v1`, so transfer and strategic-monitoring stages were not run.

See [`docs/causal-discovery-report.md`](../../docs/causal-discovery-report.md) for the complete design, figures, interpretation, limitations, hypothesis decisions, and cost accounting.

## Files

- `raw/interventions_discovery.parquet`: five-layer, four-dose discovery sweep.
- `raw/interventions_discovery.manifest.json`: exact run identity and intervention registry.
- `raw/interventions_stability.parquet`: fixed-point six-scenario stability check.
- `raw/interventions_stability.manifest.json`: exact stability-run identity.
- `summaries/`: 5,000-resample scenario-clustered summaries.
- `figures/`: publication-oriented causal and behavioral figures.
