# V4 finding: correct reasoning can impair relational self-report

## Result

Qwen3.6-27B solved the strategic/nonagentic expected-payoff decisions at 97.5%
accuracy (234/240). After a correct decision, it was asked which response had the
highest probability under its selected signal. The answer mapping used disjoint,
randomized X/Y/Z labels.

Keeping the model's own correct rationale in context made this report worse than
retaining only its final answer:

| Stage | Frozen population | Rationale | Answer only | Paired discordance | Exact p |
|---|---:|---:|---:|---:|---:|
| Discovery | 78 correct decisions | 70/78 (89.7%) | 77/78 (98.7%) | 0 vs 7 | .015625 |
| Held-out confirmation | 54 correct C-decisions | 47/54 (87.0%) | 53/54 (98.1%) | 0 vs 6 | .03125 |

The discovery contrast was one of four preregistered predicted-response controls;
its Holm-adjusted p was .0625. Every discovery discordance occurred after selecting
signal C, so the confirmation protocol froze that stratum and used only the unseen
validation and locked splits. The held-out result passed its preregistered two-sided
exact McNemar test.

## Mechanism discrimination

A post-confirmation 2x2 test crossed rationale access with response naming:

| Report condition | Indexed R1/R2/R3 | Arbitrary aliases |
|---|---:|---:|
| Answer only | 87/88 (98.9%) | 85/88 (96.6%) |
| Full rationale | 75/88 (85.2%) | 69/88 (78.4%) |

Arbitrary names did not rescue the rationale condition. They worsened it by 6.8
points (paired p=.146), while barely changing answer-only performance. The narrow
"signal C aliases to response R3" account is therefore not supported. The broader
trajectory-interference effect was strong in both naming conditions: 12 vs 0
answer-only-only discordances with indexed names (p=.000488), and 17 vs 1 with
aliases (p=.000145). These pooled mechanism statistics are descriptive because they
reuse discovery and confirmation examples.

## Candidate claim

> A model's own correct reasoning trajectory can impair later access to a simple
> relational fact that the same model reports almost perfectly when only its final
> answer is retained. Adding a second binding map amplifies rather than repairs the
> impairment.

This is consistent with self-generated relational-binding or workspace interference,
not evidence that the visible rationale is a privileged readout of a persistent
hidden belief. It directly motivates the J-space question left open by the
[workspace paper](https://transformer-circuits.pub/2026/workspace/): how relational
roles are bound and selected when the workspace contains several competing
variables. It complements work on
[belief-tracking lookbacks](https://arxiv.org/html/2505.14685v3),
[bound-entity retrieval](https://arxiv.org/html/2510.06182v2), and
[proactive interference](https://arxiv.org/html/2506.08184v1) by testing interference
from the model's own successful computation rather than an externally supplied
distractor.

## Boundaries and next experiment

- One pinned model was tested; cross-model generalization is unknown.
- The report prefix reconstructs the deterministic trajectory in context. It is not
  direct access to a persistent hidden state.
- The held-out confirmation establishes the C-decision effect, not why C was the
  vulnerable stratum.
- The alias manipulation adds mapping load, so it localizes the effect to broader
  binding/workspace interference but does not identify a circuit.

The next paid stage should be a targeted internal-state study, not a full mechanistic
sweep: use the confirmed discordant pairs plus matched correct controls, trace true
and reported response representations through the report prefix, and layerwise-patch
the answer-only report state into the full-rationale state. A rescue localized to a
narrow layer/position window would turn the behavioral result into a causal J-space
result.
