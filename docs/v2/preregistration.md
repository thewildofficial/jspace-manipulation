# Protocol V2 preregistration — H0 gate

## H0 instrumentation gate

This document and the JSON configurations in `configs/v2/` are frozen before
the first V2 GPU output is observed.

The model is `Qwen/Qwen3.6-27B` at revision
`6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`. The released lens is
`neuronpedia/jacobian-lens`, revision `qwen-n1000`, file
`qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt`.
The reference implementation and prompt protocol are pinned at Anthropic commit
`581d398613e5602a5af361e1c34d3a92ea82ba8e`.

The workspace band is selected without flexible-generalization or strategic
task outcomes. On the neutral corpus, each candidate layer is scored as
`z(excess kurtosis) - 0.5*z(J-lens/logit-lens cosine)`. Among normalized depths
0.35–0.85, the lowest-starting width-eight contiguous band with the maximum
mean score is selected.

H0 includes:

- exact transport/readout parity on fixed tensors and a real prompt;
- analytic coordinate-swap parity in a known basis;
- three finite perturbation magnitudes for a fixed J-lens direction;
- all 192 released flexible-generalization source-target trials where
  tokenization permits, under four intervention topologies.

The reference topology is workspace-band/all-prompt-positions. The earlier
exploratory comparison (from the V1 foundation protocol) is
single-layer/final-position. Baseline-incompetent items remain in
raw data and are excluded only from conditional swap success. Numeric pass
thresholds are in `configs/v2/flexible_generalization_smoke.json`.

If any H0 component fails, strategic V2 execution stops. Diagnosing or revising
the failed control becomes exploratory work and receives an append-only
decision-log entry before another run.

## Information already observed

Before this preregistration, V1 had observed: a 192/192 behavioral gate; released
lens structural/readout integrity; a six-base observational policy-family pilot;
and local causal artifacts with small policy movement, no output flips,
substantial fact-coordinate displacement, and very small output KL. The last
artifacts existed in ignored local storage rather than the initial Git commit.
They are historical V1 data, not V2 confirmation.
