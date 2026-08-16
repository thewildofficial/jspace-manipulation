# V2 final report

Status: **stopped at failed H0; no strategic V2 task was run**.

The preregistered run `b1d093ae24ea40fc8c386d3dd50a5ece` selected Qwen
layers 36–43 using the task-independent rule. Numerical parity, the analytic
swap, and the finite-direction sign criterion passed. Anthropic-style flexible
generalization did not: workspace-band/all-position clamping achieved 0/90
target top-1 outcomes among baseline-eligible trials (95% scenario-bootstrap
interval [0%, 0%]), versus 1/90 for one-layer/final-position and 25/90 for
one-layer/all-position.

The reference topology therefore missed its 25% success threshold and failed
to exceed the weakened local topology by 10 points. H0 is false. In accordance
with the preregistration, H1–H8 remain unopened.

The most informative diagnostic is that one-layer/all-position swaps worked
well for eligible country facts (24/33, 72.7%), while the full eight-layer band
worked on none. This suggests repeated symmetric swapping may undo or disrupt a
useful local edit, but the current run did not measure layer-by-layer coordinate
reconstruction and cannot choose among cancellation, reconstruction, band
mislocalization, or other explanations.

Exact effect sizes, exclusions, intervention distances, artifacts, and claim
boundaries are in `results/v2_smoke_tests/README.md` and
`results/v2_workspace_mapping/README.md`.
