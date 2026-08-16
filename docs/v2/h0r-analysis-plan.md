# H0R analysis and protocol-selection plan

## Diagnostic estimands

For each eligible source→target trial, the primary effect is

`Δ[(log p(target answer) − log p(source answer))]`.

Every diagnostic also records target top-1 conversion, positive-gain fraction,
output KL, output entropy, `||Δh||`, `ΔRMS/RMS`, source/target direction cosine,
basis condition number, and source/target coordinates immediately before and
after each hook. One-shot runs record the same coordinates at every downstream
layer.

Uncertainty uses a 2,000-draw seed-1729 bootstrap over source prompts, keeping
all targets and matched configurations from a prompt together. Diagnostic
intervals describe the burned set and are not confirmatory generalization
evidence.

## Position masks

- `final_prompt_position`;
- `argument_token_position` (token span matched to the rendered argument);
- `argument_through_end`;
- `all_prompt_positions`.

## Intervention semantics

- symmetric two-coordinate pseudoinverse swap;
- directional source suppression plus target installation, calibrated in
  natural-coordinate standard-deviation units;
- paired-donor two-coordinate patch toward the natural target-prompt state.

Identity swaps, unrelated same-category targets, fixed-seed random
two-dimensional bases, direct-answer steering, and an out-of-window layer are
reported as controls.

## Reconstruction statistic

For a one-shot source→target write at layer `l`, define oriented source evidence
`d_k = c_s(k) − c_t(k)`. Let `d_clean(k)` be the matched clean-source value and
`d_post(l)` the immediately post-write value. The return fraction is

`R_k = (d_k − d_post(l)) / (d_clean(k) − d_post(l))`,

with denominator magnitudes below `1e-6` excluded and all exclusions logged.
Report the first downstream layer where median clipped `R_k` reaches 0.25,
0.50, and 0.75. This statistic is frozen before reconstruction output is seen.

## Candidate-selection rule

Configurations are first rejected if median basis condition number exceeds 20,
absolute basis cosine exceeds 0.95 on more than 5% of trials, mean output KL
exceeds 1.0 nat, or mean `ΔRMS/RMS` exceeds 0.25. Among survivors, require
positive mean gain in at least three country functions and at least 70% positive
trial direction. Rank by the mean of within-metric ranks for: mean log-odds
gain, positive-gain fraction, target top-1 conversion, negative output KL,
negative intervention distance, and minimum function effect. Ties within 0.02
rank units choose fewer layers, then the narrower position mask, then lower
alpha. Exactly one winner is committed before H0R-C.
