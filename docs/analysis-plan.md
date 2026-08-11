# Analysis, artifacts, and figure plan

## Artifact rules

Raw numerical outputs are append-only JSONL or Parquet files. Plotting never launches a model and never mutates raw results. Each row carries `run_id`, `created_at`, seed, resolved revisions, GPU, elapsed seconds, and schema version. A `manifest.json` beside each run records the command and hashes of inputs.

The default raw tables are:

- `behavior.jsonl`: one row per prompt condition.
- `lens_readout.parquet`: one row per scenario × layer × position × ranked token.
- `residual_summaries.parquet`: frozen scalar coordinates and norms; full activations are optional cached arrays and are not required for figures.
- `interventions.parquet`: one row per scenario × layer × alpha × intervention/control.

## Behavioral schema

Required fields:

```text
scenario_id, family, template_id, split, policy_style,
world_state, policy, expected_report, prompt,
logp_A, logp_B, predicted_report, correct,
literal_logodds_A_over_B, truth_aligned_score, policy_following_score
```

Rows are paired by `scenario_id + policy_style + world_state`; all bootstrap resampling uses `scenario_id`.

## Intervention schema

Required fields:

```text
scenario_id, family, split, world_state, policy, layer, layer_fraction,
alpha_sd, intervention_type, direction_name, control_seed,
baseline_literal_logodds, intervened_literal_logodds,
delta_literal_logodds, delta_policy_score,
baseline_fact_score, intervened_fact_score, delta_fact_score,
delta_residual_rms_ratio, output_kl, output_entropy
```

`intervention_type` is one of `steer`, `ablate`, `swap`, `direct_answer`, `direct_fact`, `random`, `unrelated`, or `lexical`.

## Figures and their interpretation

### Behavioral gate

A point-range plot shows accuracy and policy-following margin for each world × policy cell, faceted by policy style. This catches literal-answer bias and lexical dependence. The figure is a gate, not evidence for a latent mechanism.

### J-space trajectory

Plot standardized, held-out concept-family scores over normalized layer depth for fact, policy, and imminent answer. Lines are scenario means; bands are scenario-bootstrap intervals. Top-k rank trajectories are supplementary because ranks can exaggerate small logit changes.

### Layer-by-position map

Representative examples are chosen by a preregistered rule (nearest to median causal effect), not visual appeal. Show full prompt tokens and selected concept ranks. Qualitative maps illustrate but do not establish prevalence.

### Causal layer sweep

Plot paired `delta_policy_score` over depth with the candidate swap/steer, direct-answer, unrelated, and random controls. Include simultaneous exploratory bands. The confirmatory marker highlights only the frozen layer band.

### Dose response

Plot alpha in empirical coordinate SDs against behavior, fact score, and residual-distance ratio. The primary region ends at the natural-distribution threshold; extrapolative points use open markers.

### Two-by-two sign-flip matrix

For the same conceal intervention, plot `delta_literal_logodds` separately for fact A and fact B. A policy effect has opposite signs. An answer-A direction has positive signs in both cells. This is the most interpretable discriminator between mapping-level and token-level steering.

### Behavior versus fact preservation

Each point is a scenario's behavior change against fact-readout change. Show the preregistered fact-equivalence band. Large horizontal movement near zero vertical movement is the desired regime; the plot must include direct-fact and random calibration points.

### Transfer

Plot frozen-effect estimates by discovery, validation, and held-out family. Never pool discovery and test into the headline number.

## Statistical details

- Use 5,000 scenario bootstrap resamples for final intervals and 1,000 for development plots.
- Report effect sizes in log-odds and probability-point changes; avoid accuracy alone.
- For layer exploration, use max-|t| or bootstrap simultaneous bands. Do not report the best layer's pointwise interval as confirmatory.
- Define a smallest effect size of interest before test: suggested policy change ≥ 0.5 log-odds and fact change within ±0.2 standardized probe units. Calibrate these values on development stability, then freeze them.
- Report every attempted concept family and control, including failed directions.
- Treat scenario family as a fixed transfer axis; with only a few families, a random-effects claim is not warranted.

## Reading negative findings

- A J-lens token may be a downstream correlate of policy, not a causal input.
- Failure of a single token does not rule out a distributed policy representation.
- Failure on indirect wording suggests lexical echo.
- Failure across output symbols suggests answer-template dependence.
- Large fact shifts indicate entanglement or an invalid fact metric.
- Effects only at extreme alpha are off-distribution steering, not clean causal rewriting.
