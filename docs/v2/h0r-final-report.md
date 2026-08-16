# H0R final report

Status: **H0R-B complete; candidate frozen; prospective controls unopened**.

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

H0R-C and H0R-D remain prospective and unopened. This report will be finalized
only after their frozen-gate evaluations.
