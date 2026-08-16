# V2 H0 instrumentation and flexible-generalization gate

## Outcome

**H0 failed. No strategic-reporting experiment is licensed or was run.**

Numerical transport/readout parity, the analytic coordinate-swap test, and the
predeclared finite-direction sign check passed. The required causal positive
control did not: the reference topology (workspace band, all prompt positions)
produced 0/90 target-answer top-1 successes on baseline-eligible trials.

| Topology | Eligible | Target top-1 rate (95% scenario bootstrap) | Target-vs-source log-odds gain (95% CI) | Mean `ΔRMS/RMS` |
|---|---:|---:|---:|---:|
| One layer / final position | 90 | 1.1% [0.0, 3.4] | 0.82 [0.49, 1.22] | 0.091 |
| One layer / all positions | 90 | 27.8% [14.6, 42.7] | 5.10 [3.12, 7.41] | 0.075 |
| Band / final position | 90 | 0.0% [0.0, 0.0] | 0.22 [0.13, 0.33] | 0.047 |
| Band / all positions | 90 | 0.0% [0.0, 0.0] | 0.58 [0.36, 0.87] | 0.060 |

The preregistered reference success threshold was 25%, with at least a
10-percentage-point advantage over one-layer/final-position. The observed
reference rate was 0%, 1.1 points below that weakened comparison. Although its
mean log-odds gain exceeded the separate +0.5 threshold, the behavioral and
topology criteria failed.

The unexpected useful signal is the one-layer/all-position condition. It
succeeded on 24/33 eligible country trials (72.7%) but persistent bandwise
clamping succeeded on 0/33. This is diagnostic evidence that topology matters,
but in the opposite direction from H8 and the H0 prediction. It does not rescue
the gate post hoc.

## What was tested and why

The run tested exact local/upstream transport parity, direction construction,
an analytically known coordinate swap, finite perturbations, and all released
flexible-generalization trials that had valid one-token arguments/answers for
Qwen. The causal prompts came byte-for-byte from Anthropic's pinned released
JSON. Four matched topologies isolated layer persistence from position extent.

Before this run, V1 had already shown behavioral competence, released-lens
integrity, an observational policy-family signal, and a weak local causal
result with no flips. Only H0 was confirmatory here.

## Gates, exclusions, and controls

- Baseline task accuracy was 52.9% over retained source-target rows (90/170 per
  topology eligible). Baseline failures remain in raw data and are excluded
  only from conditional swap success.
- Fourteen logged tokenization exclusions account for released answers without
  an accepted one-token continuation; 170 of 192 trials remained per topology.
- The independently mapped Qwen band was layers 36–43; the single-layer
  comparison used its center, layer 40.
- Synthetic swap maximum error, random-tensor transport error, prompt readout
  error, and direction-formula error were all exactly zero at stored precision.
- The two larger finite perturbations moved the target token in the predicted
  sign; the smallest was below bfloat16 readout resolution. Full-vocabulary
  delta cosine was weak (−0.024, 0.002, 0.183), so this check supports only its
  narrow preregistered sign criterion.
- Intervention distance is reported above. Target coordinates were explicitly
  exchanged at every requested hook. Downstream reconstruction and independent
  task-state preservation were not measured in H0 and must not be inferred.

## Reproducibility

- Command: `modal run modal_v2.py::h0`
- Run: `b1d093ae24ea40fc8c386d3dd50a5ece`
- Code commit: `ed187cba8e2429ce86ba305f467ada30f8be1768`
- Tree: dirty because a pre-existing unrelated untracked `.prismor/` directory
  was present; tracked experimental files matched the commit.
- Workspace config SHA-256: `01db216b10c465455170291565baf222d47bc3748e6ca5b1f2126964ea337db7`
- Flexible config SHA-256: `97de7c122da5b3c6db56972716a84f30689658529d00ad07770ecf198f6fc872`
- Model revision: `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`
- GPU: NVIDIA A100-SXM4-80GB
- Metered elapsed time: 151.27 seconds; conservatively recorded cost: $0.1484

The first invocation's remote return could not be deserialized locally and was
logged separately as a conservative $0.1766 failed-run estimate. No metric from
that invocation was inspected.

## Claim boundary

Licensed: local transport and swap formulas match the pinned implementation and
analytic expectations; a single-layer/all-position country swap can causally
redirect many eligible Qwen answers; repeating the symmetric swap through the
selected band suppresses rather than strengthens that effect.

Not licensed: V2 instrument validation, latent-knowledge monitoring, strategic
misreporting claims, truth/report dissociation, causal latent-truth swapping, or
cross-task transfer.
