# V5-RBG-6 analysis plan: endogenous full action trajectory

**Status:** frozen after RBG-4 and before any full-trajectory model execution.

This study resolves [issue #11](https://github.com/thewildofficial/when-words-override-consequences/issues/11).
RBG-3's causal rehearsal placed oracle-correct reports into the context. It did
not ask the model to produce those reports and then make its choice while seeing
its own answers. RBG-6 runs that missing trajectory.

This is a frozen follow-up on the already observed RBG-4 dataset, not an
independent replication. The new full-trajectory outcomes were unobserved when
this plan and its thresholds were fixed.

## Design

The source is the immutable RBG-4 run
`7c06999f4ad54d06b4454584e04faf9d` (file SHA-256
`4e7ba1d0dc16e1fe231d7fe312e77dbede14e33e5db2eca2c6e030ed9237b4f9`).
All 384 opposed-goal assertion and opaque-token contexts are included. Each is
run in both report orders, `A -> B` and `B -> A`, for 768 trajectories.

The model first reports the consequence of one option as `X` or `Y`. That exact
answer is retained as an assistant turn. It then reports the other option, sees
both answers in context, and chooses `A` or `B`. `X` and `Y` are explicitly
mapped to the two literal outcomes in each report question.

Four arms are compared:

1. **Direct:** the immutable RBG-4 action without report turns.
2. **Self-generated:** both model-generated reports remain in the action trajectory.
3. **Oracle replay:** the same trajectory with both correct report tokens supplied.
4. **Swapped replay:** the same trajectory with both report tokens inverted.

Oracle and swapped replay distinguish report-generation errors from sensitivity
to visible report content. Report order tests recency. Matched opaque-token rows
test whether the trajectory generally damages action selection.

## Frozen endpoints

The primary population is the RBG-2-like cell: strategic, opposed, assertion,
redundant history, and prose mapping (`n=48` after report-order duplication).
The primary outcome is the conjunction “both self-generated reports correct and
final self-generated action wrong.” Report the count, rate, and exact 95%
binomial interval. A persistent full-trajectory signature requires at least 90%
report accuracy and at least 25% conjunction prevalence.

A rehearsal-rescue signature requires self-generated action accuracy to exceed
direct accuracy by at least 25 points with two-sided exact base-game clustered
sign-flip `p<.05`. The two report orders are robust if their action accuracies
differ by at most 10 points. The matched opaque control must retain at least 90%
self-generated action accuracy.

Oracle and swapped replay are descriptive causal transcript controls. All eight
opposed assertion frame x history x mapping cells are reported.

## Compute placement

Dataset construction, prompt rendering, tokenization, one-token validation,
statistics, and report generation run on the GitHub Actions CPU runner. Every
possible report branch is tokenized before Modal starts. The Modal function
contains only model loading, batched GPU forward passes, minimal routing between
pre-tokenized branches, and result serialization. There is no Modal CPU
preflight or statistical analysis while an accelerator is reserved.

The A100-80GB function has a 1,200-second hard timeout and a USD 2 study ceiling.
Identical action branches are de-duplicated before inference.

## Claim boundary

This experiment can show whether explicit, self-generated consequence reports
control later behavior in this task. It cannot establish deception,
consciousness, a stable preference, or a particular internal mechanism.
