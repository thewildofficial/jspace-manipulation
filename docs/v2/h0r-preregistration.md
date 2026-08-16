# H0R preregistration: Qwen causal-instrument recovery

H0R resolves the failed V2 causal-instrument gate without retroactively changing
H0. The failed result is preserved at commit
`5c7379f4dc601f8eb47a1487aad9d7bf76186cc7` and remains the core verdict in
`results/v2_smoke_tests/README.md`.

## Information burned before H0R

- workspace-band/all-position target top-1: 0/90;
- single-layer/final-position target top-1: 1/90;
- single-layer/all-position target top-1: 25/90;
- country-only single-layer/all-position target top-1: 24/33;
- operational readout band: layers 36–43; center layer: 40;
- exact numerical transport/readout parity, analytic swap parity, and the
  preregistered finite-direction sign check passed.

The released Anthropic flexible-generalization set is burned and may be used
only for exploratory topology diagnosis. Fresh argument and computed-
intermediate controls will be generated, tokenized, hashed, and committed
before the first diagnostic GPU run. Their model behavior and intervention
outcomes remain unopened until a single candidate protocol is frozen.

## Recovery phases

1. **H0R-A:** record basis cosine, condition number, pre/post coordinates,
   next-layer coordinates, and fp32 coordinate-exchange error.
2. **H0R-B:** on burned country trials, sweep layers 28–52, four position masks,
   strengths 0.25/0.5/1/1.5/2, cumulative and odd/even layer sets, intervention
   semantics, one-shot reconstruction, and matched controls.
3. Freeze exactly one protocol using the rule in the analysis plan and commit
   it before opening fresh controls.
4. **H0R-C:** prospectively evaluate the frozen protocol on fresh argument
   substitution.
5. **H0R-D:** if C passes, prospectively evaluate the same protocol on a
   computed intermediate.

The V2 causal branch is unblocked only if H0R-C and H0R-D both pass. No H0R
result changes the original H0 verdict.

## Prospective gates

H0R-C is valid only when baseline accuracy is at least 80% overall. It passes
only if the scenario-bootstrap 95% interval for mean target-vs-source log-odds
gain excludes zero, mean gain is at least +0.75, at least 70% of eligible trials
move positively, at least 20% make the target answer top-1, and the semantic
gain exceeds matched random/unrelated control by at least +0.5. Output KL,
residual displacement, basis cosine, and condition number must remain inside
the diagnostic protocol's frozen validity domain.

H0R-D passes only if the bootstrap interval for computed counterfactual gain is
above zero, mean gain is at least +0.5, at least 65% of eligible trials move
positively, semantic intervention exceeds random and unrelated J-space
controls, and comparable direct-answer steering does not reproduce the same
pattern.

No threshold, layer, mask, alpha, operation, normalization, or tokenization
rule may change after the candidate protocol is committed.
