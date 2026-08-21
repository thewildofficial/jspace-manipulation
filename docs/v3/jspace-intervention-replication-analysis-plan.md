# V3-JI1 analysis plan

## Confirmatory hypotheses

**H1:** At alpha 1, native-raw semantic swaps have positive target-vs-source
log-odds effects and exceed both paired random-displacement and unrelated-
semantic controls.

**H2:** The paper-style clean source-loading statistic positively predicts the
source-prompt mean alpha-1 semantic effect. Final-position-only loading is a
mandatory mechanistic robustness analysis.

Alpha 0.5/2 dose response, top-1 conversion, distortion, topology, vector
normalization, categories, and reconstruction are secondary or diagnostic.

## Analysis unit

For every `(category,function,source_argument)` cluster, average the three target
swaps separately by alpha and control. Use 2,000 cluster-bootstrap draws with
seed 1729. Report 48 expected fresh source-prompt clusters, 144 raw semantic
target swaps per alpha, and 12 distinct source concepts if the planned 3x4x4
corpus freezes successfully.

A source-concept-clustered sensitivity interval is required. It is expected to
be imprecise with only 12 concepts and constrains claims of concept-level
generalization.

Before Phase D, use the burned cluster variance to simulate detection
probability for true mean effects of 0.25, 0.5, 1.0, and 2.0. If the probability
of satisfying H1 at the minimally meaningful +0.5 effect is below 80%, describe
the fresh study primarily as estimation/characterization; do not treat failure
to cross a tier as strong evidence of absence.

## Required summaries

- semantic mean Delta and 95% CI at each alpha;
- positive trial fraction and target-top-1 conversion;
- paired semantic-minus-random and semantic-minus-unrelated means/CIs;
- KL(clean || patched), entropy, residual displacement, cosine, and condition
  number distributions including median/p90/p95/max;
- Spearman loading correlations with cluster-bootstrap intervals;
- per-category and per-function effects;
- raw-versus-normalized and band-versus-single-layer burned comparisons.

No post-hoc configuration becomes the headline result.
