# Qwen3.6-27B observational J-lens pilot

This six-base, 48-prompt discovery pilot asks whether balanced toy prompts contain a consistent J-lens signal for reporting policy and world fact before any intervention is attempted.

This is exploratory discovery data. It was used to assess candidate measurements and cannot serve as held-out confirmation.

## Design

Every base prompt is crossed with:

- fact `A` or `B`;
- identity/reveal or swap/conceal policy; and
- named COPY/FLIP or unlabeled R0/R1 wording.

At the final prompt boundary, the official 1,000-prompt J-lens records all 63 fitted layers. Each record contains the top ten vocabulary tokens and fixed seed-family scores for fact, reveal, conceal, lexical-control, and unrelated-control tokens.

The policy coordinate is `conceal-family score − reveal-family score`. The plotted policy contrast pairs otherwise matched prompts and computes `conceal prompt − reveal prompt`. Positive values mean the internal readout moves in the expected semantic direction.

## Exploratory findings

1. The policy contrast is near zero through early depth, rises around normalized depth 0.55, and remains positive through the late workspace range.
2. The unlabeled R0/R1 prompts show the same broad rise as named COPY/FLIP prompts. Their largest mean paired contrast is +2.23 at layer 44. This makes pure lexical echo of the words COPY/FLIP an insufficient explanation.
3. Named prompts show a very large spike in layers 59–62 (mean +16.64 at layer 59), while the unlabeled condition does not. That late spike is likely label/output preparation and is not a good causal target by itself.
4. The literal A-versus-B seed-family fact coordinate is weakly state-sensitive in the middle layers but reverses and becomes unstable late. It should not be used as the sole “fact preserved” metric.
5. These observations justify the preregistered independent fact probe and favor the shared middle-layer band—approximately layers 43–47—for causal discovery over the visually largest late-layer peak.

The finding is interesting but not yet novel evidence of deception. It establishes a candidate abstract policy signal that survives removal of semantic rule labels. We still need a causal, world-state-conditioned sign flip, independent fact preservation, controls, and held-out transfer.

## Figure

![Paired policy and fact trajectories](figures/lens_readout_trajectories_27b.png)

Thin lines are individual base prompts; thick lines are means. They show discovery variability rather than confirmatory confidence intervals.

## Files

- `raw/lens_readout_27b.jsonl`: every prompt × fitted-layer record with top tokens and family scores.
- `raw/lens_readout_27b.manifest.json`: exact model/lens identities, token sets, and run metadata.
- `summaries/lens_policy_trajectory_27b.csv`: paired scenario-level policy contrasts.
- `summaries/lens_fact_trajectory_27b.csv`: paired scenario-level fact contrasts.
- `figures/lens_readout_trajectories_27b.png`: interpretable trajectory figure generated from the raw table.
