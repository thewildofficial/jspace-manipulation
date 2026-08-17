# Stage 1 preregistration: latent state–report dissociation

## Scope and historical boundary

Stage 1 is an observational study of `Qwen/Qwen3.6-27B` under the pinned
Jacobian Lens. The original H0 causal-control gate and the H0R prospective
counterfactual-replacement gate remain failed historical results. Stage 1 does
not repair those gates, optimize an intervention, or make a causal claim.

The question is whether a controlled task-relevant state, (X), remains
prospectively recoverable at the final prompt position when a neutral reporting
policy, (U), selects a different literal report, (R):

\[
R=f(X,U),\qquad U=T: R=X,\qquad U=M: R=\pi(X).
\]

Here \(\pi\) is one of three counterbalanced cyclic derangements over four
candidate labels. Primary prompts do not use language about deception,
dishonesty, concealment, misleading, or output flipping.

## Frozen model and readout

- Model: `Qwen/Qwen3.6-27B`
- Model/tokenizer revision: `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`
- Lens artifact: `neuronpedia/jacobian-lens`, revision `qwen-n1000`
- Lens file: `qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt`
- Lens code: Anthropic commit `581d398613e5602a5af361e1c34d3a92ea82ba8e`
- Primary position: the final prompt token, before any answer token is generated
- Primary band: layers 36–43 inclusive, inherited from Stage 0

No generated answer token enters the primary monitor.

## Dataset

Stage 1A supplies the private state explicitly. Stage 1B requires it to be
computed from evidence. Each substage has 12 distinct generator families,
frozen as six discovery, three validation, and three locked-test families.
Every family contains ten independent base scenarios. Every base scenario is
rendered in all eight state-by-policy cells, producing 960 prompts per substage
and 1,920 prompts in total.

State labels come from six counterbalanced four-label vocabularies. The pinned
tokenizer must represent every candidate as one distinct continuation token in
the frozen answer format. Tokenization failure is a deterministic exclusion;
no row may be excluded for a weak or inconvenient mechanistic result.

The generator seed, exact family assignments, codebooks, prompt hashes, and
metadata schema are frozen in `configs/v2/stage1.json` and
`src/jspace_policy/stage1.py`. The tokenizer-enriched rendered corpus is
content-hashed and committed before behavior is opened.

## Behavioral gate

Behavior is evaluated without saving activations or J-space output. Stage 1A
and Stage 1B must each reach at least 90% exact expected-token accuracy across
discovery and validation, with no state-by-policy cell below 85% and no family
below 80%. A failure permits redesign using behavioral evidence only and an
append-only decision entry. Once mechanistic output for a family is observed,
that family is burned for task design.

The locked-test families remain behaviorally unopened until the analysis and
probe protocol are frozen. Their same thresholds are validity gates; failure
makes the corresponding mechanistic result invalid rather than negative.

## Confirmatory evidence

For candidate state token \(x_i\), let \(s_\ell(x_i)\) be its raw pinned
J-lens vocabulary score at the final prompt position. Define

\[
K_\ell=s_\ell(X)-\frac{1}{3}\sum_{x_j\ne X}s_\ell(x_j),
\]

\[
Q_\ell=s_\ell(R)-\frac{1}{3}\sum_{x_j\ne R}s_\ell(x_j),
\qquad D_\ell=K_\ell-Q_\ell.
\]

Band quantities are arithmetic means across layers 36–43. Positive transformed
\(\bar K\) is the primary latent-state-retention endpoint.

To measure report preparation while holding the candidate fixed, define
\(M_\ell\) as evidence for \(\pi(X)\) in both matched policies. The policy
effects are

\[
\Delta_M=\bar M_{U=M}-\bar M_{U=T},\qquad
\Delta_K=\bar K_{U=M}-\bar K_{U=T},
\]

and the state-versus-report interaction is \(\Delta_M-\Delta_K\). This fixed-
candidate contrast operationalizes the policy-sensitive report hypothesis;
comparing \(Q\) directly across policies would change which token is scored.

Primary uncertainty is a 2,000-draw bootstrap with seed 981723. The resampling
unit is the base scenario, retaining all eight factorial rows. Report pooled
and family-specific effects. A substage meets the confirmatory dissociation
criterion only when locked transformed \(\bar K\) has a 95% interval above
zero, at least two of three locked families have positive mean effects, and an
independent state probe is above four-way chance with its interval above 0.25.
Stage 1B must pass separately for language about inferred state.

## Independent probes and baselines

Regularized multinomial linear probes are trained only on discovery families.
Layer and inverse regularization strength are selected on validation families
from layers 36–43 and C values 0.01, 0.1, 1, and 10. Ties choose the lower layer
then lower C. Separate state and report probes are fit, along with state probes
trained on truthful examples and tested on transformed examples and vice
versa. Frozen coefficients, scalers, layers, and hyperparameters are committed
before opening the locked test.

Probe metrics are four-way accuracy, macro one-vs-rest AUROC, negative
log-likelihood, and expected calibration error. Baselines are final output
logits and a conventional logit lens at the same position/layers. J-space need
not outperform every baseline, but any advantage is reported quantitatively.

## Locked-test and claim boundary

Before opening the test, a freeze commit records dataset/config hashes, test
families, model/lens revisions, scores, probe artifacts, bootstrap settings,
plots, tests, and claim ladder. The locked test is executed once. Infrastructure
repairs made before scientific output materializes are logged append-only.

Stage 1 is not evidence of deception, intent, consciousness, scheming, a
general lie detector, or causal use of a representation. A positive Stage 1B
licenses only the claim that, in these controlled tasks, inferred task state
remained prospectively decodable before a conflicting policy-selected report.
Stage 2 remains closed until the Stage 1 report is committed.

