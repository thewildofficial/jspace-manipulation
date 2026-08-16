# H0R final report

Status: **H0R-B complete; first H0R-C corpus invalid; intervention unopened**.

The original H0 remains failed. The exploratory burned-set diagnostic selected
layers 36--42, argument-through-end positions, a pseudoinverse coordinate
operation at alpha 0.5, and unit-normalized per-layer J-lens directions. The
selection commit includes the complete raw results, summaries, required
figures, and `configs/v2/h0r_candidate_protocol.json`.

The diagnostic result was mean target-vs-source log-odds gain +4.34 (95%
scenario-bootstrap interval +3.62 to +5.21), 100% positive direction, 1/33
target top-1, output KL 0.227 nats, and mean delta-RMS ratio 0.017. Identity and
random-basis controls were null. Full interpretation is in
`docs/v2/h0r-diagnostic-report.md`.

The first locked H0R-C corpus achieved only 60.8% baseline accuracy, below the
80% behavioral gate. The runner stopped before intervention inference, so this
does not test the causal protocol. Following the preregistered rule, a simpler
replacement argument corpus will be frozen and baseline-checked. The protocol,
validity limits, and causal pass thresholds remain unchanged. H0R-D remains
unopened.
