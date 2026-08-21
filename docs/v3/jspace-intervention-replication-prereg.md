# V3-JI1 preregistration: replication and characterization of J-space interventions

**Branch:** `research/jspace-intervention-replication`  
**Study:** `V3-JI1`  
**Status:** frozen before new intervention results

## Question and historical boundary

V3-JI1 asks what causal effects Jacobian-Lens coordinate interventions support
on the pinned Qwen stack, and under what conditions. It does not reclassify the
failed H0R-C result. H0R-C remains a prospective failure of its frozen 20%
target-top-1 criterion despite its positive directional effect.

The study distinguishes directional causal influence, semantic selectivity,
low average distortion, behavioral semantic substitution, and reliable state
replacement. These are not interchangeable claims.

## Pinned stack and intervention

- model: `Qwen/Qwen3.6-27B` at
  `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`;
- lens: `neuronpedia/jacobian-lens`, revision `qwen-n1000`, official 1,000-prompt
  Qwen3.6-27B artifact;
- reference lens code: commit `581d398613e5602a5af361e1c34d3a92ea82ba8e`;
- BF16 inference, greedy next-token scoring, thinking disabled;
- primary workspace layers 36--43, all prompt positions.

For raw J-lens vectors `V=[v_s,v_t]`, coordinates are `c=pinv(V)h` and the
intervention is

```text
h' = h + alpha * V * (swap(c) - c).
```

All vector construction, pseudoinverses, coordinates, deltas, and diagnostics
are computed in FP32 before casting the patched residual back to BF16.
`alpha={0.5,1,2}` respectively equalizes, swaps, and overshoots the two local
coordinates. Native raw vectors are primary. Unit-L2 vectors appear only in the
burned historical comparison.

## Phases

1. **A -- numerical validation.** Analytic coordinate equations, identity,
   orthogonal preservation, position masks, clean-hook parity, and layer-index
   parity must pass.
2. **B -- burned engineering replication.** The released countries/months/
   animals/numbers family may validate implementation, estimate cost, compare
   topology, and compare raw versus normalized vectors. It cannot support a
   prospective claim.
3. **C -- behavior-only freeze.** The ordered candidate pool is evaluated using
   tokenizer checks and clean inference only. Code selects the first feasible
   4-argument x 4-function block per category. No manual repair is allowed.
4. **D -- prospective validation.** After the selected corpus and its hash are
   committed, run the three frozen alphas and all controls once. No rescue sweep.

## Fresh corpus selection

`configs/v3/jspace_interventions/fresh_candidates.json` contains ordered
overcomplete pools for countries, months, and number words. A cell is feasible
only if:

- the contextual argument span is exactly one suitable token;
- appending the answer adds exactly one contextual continuation token;
- the four answers within a retained function have distinct token IDs;
- the clean model's top-1 continuation is the source-consistent answer.

For each category, the selector enumerates 4-argument combinations and then
4-function combinations in JSON order and retains the first all-feasible block.
The confirmatory population is therefore explicitly conditional on pinned-model
competence and tokenization validity.

## Controls

- **Identity:** source-to-source through the same hook path. It is a numerical
  no-op and is excluded from conditioning summaries because `[v_s,v_s]` is rank
  deficient by construction.
- **Random displacement matched:** at every trial/layer, generate a deterministic
  Gaussian 2D basis, project it out of the semantic span, QR-orthonormalize it,
  perform the same swap, and rescale each patched activation's delta to exactly
  match the semantic delta L2 norm at that hook.
- **Unrelated semantic:** for each source, cyclically map its three ordered
  targets to the next target. This is balanced and never equals source or target.

On the burned set only, the study also excludes the final prompt position and
compares a single center layer with the full band. These diagnostics cannot
select a new fresh configuration.

## Outcomes and inference

For target answer `y_t` and source answer `y_s`, define

```text
M = log p(y_t) - log p(y_s)
Delta = M_patched - M_clean.
```

The primary condition is native-raw, alpha 1. The uncertainty unit is the clean
source prompt `(category,function,source_argument)`. All three target swaps and
their alphas/controls remain together in 2,000 fixed-seed cluster-bootstrap
draws. Raw target-swap counts and cluster counts are both reported.

For both controls, calculate paired contrasts
`Delta_semantic - Delta_control`. Tier 1 requires a positive semantic 95% CI,
at least 70% positive raw semantic trials, positive 95% CIs for both paired
contrasts, and a point mean of at least +0.5 for each contrast.

Tier 2 adds the historical mean limits: KL(clean || patched) at most 1 nat,
mean residual delta-RMS/RMS at most 0.25, median condition number at most 20,
and at most 5% of bases with absolute cosine above 0.95. Meeting these limits
supports only the phrase **low distortion on average**. Median, p90, p95,
maximum, and threshold-exceedance fractions are mandatory so tail failures are
not hidden.

Tier 3 additionally requires target-answer top-1 conversion of at least 20%, at
least a 10-point advantage over each control, and positive mean effects in at
least 75% of retained functions.

## Loading hypothesis

The paper-style loading statistic averages source-vector cosine over argument
and final prompt positions across layers 36--43. A mandatory robustness measure
uses only the final prompt position. If only the paper-style metric predicts
effect, the conclusion is limited to replication of that operational measure.
Claims about persistent workspace recruitment require agreement from the
final-position-only measure.

## Permitted conclusions

Depending on results, V3-JI1 may support selective directional steering, low
average-distortion steering, or behavioral substitution within this exact
model/task/intervention validity domain. It cannot establish complete J-space
cognition, concept absence from failed swaps, consciousness, intentions,
deception, arbitrary semantic writing, or a general debugger-like API.

This is a prospective within-family model-transfer replication and instrument
characterization, not a line-for-line replication of private Claude runners.
