# V2-E1 Strategic Workspace Atlas replication freeze

Status: **frozen after open-atlas inspection and before locked behavior**.

## Open evidence used for selection

The 144-row open corpus passed its behavior gate. The all-layer mechanistic run
then produced 144 final-position readouts, 12 deterministic deep traces, and
4,572 residual/J-space/output decoder evaluations. All selections below are
post-hoc with respect to that open atlas and prospective only with respect to
the unopened locked rendering family.

The open atlas suggested a specific selective-routing account:

- a generic optimization concept (`optimal`/`optimize` variants) appeared in
  every Kuhn and signaling row at layer 43, before conventional-logit-lens
  action commitment;
- exact strategy class was strongly recoverable from residual activations in
  the five exact games, while a top-50 J-space rank-feature decoder and final
  legal-action logits were generally much weaker; and
- Kuhn and signaling output commitment was late, with censored medians at
  layers 62 and 60.5 respectively.

The Chameleon `safe`/`risk` token pattern is not advanced because it occurred
only under the discovery rendering and did not survive the open validation
template. No visually selected deep trace is advanced.

## Frozen endpoints

### H1: generic optimization workspace

At layer 43, normalize decoded top-50 strings by stripping whitespace and
lowercasing. For each locked Kuhn and signaling row, find the best rank among
`optimal`, `optimize`, `optimally`, and `optimization`.

H1 passes separately for each game if the token-family presence rate is at
least 90%, median best rank is no worse than 35, and its layer precedes the
row's conventional-logit-lens commitment layer in at least 80% of rows.
Rows uncommitted through layer 62 are censored at layer 63.

### H2: strategy-routing dissociation

For each exact game, train regularized strategy classifiers on the combined
discovery and validation rows and evaluate once on locked rows. Use the frozen
residual and J-space layers in the machine freeze. The output baseline uses the
three final legal-action log probabilities.

H2 passes if mean locked residual balanced accuracy across the five games is at
least 0.70, its mean advantage over the stronger per-game J-space/output
baseline is at least 0.15, and residual accuracy exceeds the stronger baseline
in at least four games.

The J-space classifier uses reciprocal-rank features from the frozen top-50
token inventory. This tests vocabulary-visible inventory, not the paper's
formal sparse nonnegative decomposition.

### H3: same-action Kuhn strategy

Restrict to rows whose solver-optimal action is `A=BET`; these contain bluff,
thin-bet, and value-bet cases with identical external action. Train on both open
splits and test the locked split at layer 42.

H3 passes if residual balanced accuracy is at least 0.65 and exceeds the
stronger of the layer-42 J-space decoder and output-logit baseline by at least
0.15.

### H4: late action commitment

Commitment is the earliest fitted layer after which the same conventional-
logit-lens legal action remains top-ranked through layer 62 and agrees with the
final output legal choice. Uncommitted rows are assigned layer 63.

H4 passes separately for Kuhn and signaling if locked median commitment layer
is at least 58.

## Locked execution

The locked runner may recompute deterministic open activations solely to train
the frozen decoders, then evaluates them on the locked rendering family. Only
locked top-token rows are retained in the returned atlas. All fitted lens
layers are still collected because H1 and H4 require fixed-layer and trajectory
readouts; only the layers named above enter H2/H3.

The locked behavior gate is unchanged: 98% formatting compliance, 70% pooled
exact-game accuracy, and 50% accuracy in every exact game. Failure leaves all
locked mechanistic endpoints unopened.

The exact machine rules are in
`configs/v2/workspace_atlas/replication_freeze.json`. No endpoint is causal,
and the four endpoints are reported separately without an omnibus study-pass
label.
