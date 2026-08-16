# Stage 1 machine analysis plan

## Schemas and signs

Every rendered row contains `condition_id`, `substage`, `split`, `family`,
`base_scenario_id`, `world_state_id`, `policy_id`, `codebook_id`, four ordered
candidate strings and token IDs, `true_state`, `transformed_state`,
`expected_report`, exact prompt/token IDs, prompt hash, and generation seed.

For a four-vector of scores `z` and target index `i`, evidence is exactly
`z[i] - mean(delete(z, i))`. `K` targets `world_state_id`; `Q` targets the
expected-report index; `M` always targets the transformed-state index.
Band scores are unweighted layer means. No normalization is applied to J-lens,
logit-lens, or output-logit candidate scores before evidence contrasts.

## Behavioral summaries

`correct := top1_token_id == expected_report_token_id`.
`formatting_compliant := top1_token_id in candidate_token_ids`. Gates are
computed separately for Stage 1A and Stage 1B on the opened split set. Overall,
all 4×2 state-policy cells, and all included families must pass their respective
thresholds in `configs/v2/stage1.json`.

## Probe fitting

Inputs are float32 residual vectors at the final prompt position. For each
candidate layer, features are standardized using discovery-only mean and scale.
A multinomial L2 logistic regression with fixed seed 981723 and maximum 1,000
iterations is fit for each frozen C. State labels are `world_state_id`; report
labels are the candidate index of `expected_report`.

Selection maximizes validation accuracy, with ties resolved by lower layer then
lower C. State and report probes select independently. Cross-policy selection
maximizes the mean of truthful-trained/transformed-validation accuracy and
transformed-trained/truthful-validation accuracy, with the same tie break.
Neither validation labels nor vectors are included in final fitted weights.

## Bootstrap estimands

Each draw samples `base_scenario_id` values with replacement within substage.
All rows belonging to a sampled scenario are repeated together.

- Retention: mean band `K` over transformed (`policy_id == M`) rows.
- Report preparation: within each scenario/state pair, transformed-policy
  `M` minus truthful-policy `M`, then the mean of paired differences.
- State policy effect: analogous transformed-policy `K` minus truthful-policy
  `K`.
- Interaction: paired report-preparation difference minus paired state-policy
  difference.
- Probe accuracy: mean exact correctness over rows; chance reference is 0.25.

Percentile intervals use quantiles 0.025 and 0.975 over exactly 2,000 draws.
Family direction uses the unbootstrapped family mean transformed `K > 0`.

## Output files

Raw remote returns are immutable JSON files under `results/v2_stage1/raw/`.
The deterministic analysis writes CSV summaries and figures under the adjacent
`summaries/` and `figures/` directories and regenerates
`docs/v2/stage1-final-report.md`. No reported number is entered manually.

