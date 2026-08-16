# H0R final report

Status: **H0R-C failed; causal V2 branch remains blocked**.

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
80% behavioral gate. Its runner stopped before intervention inference. A
simpler replacement was then frozen using baseline competence only; the
protocol, validity limits, causal thresholds, and controls were unchanged.

The replacement passed baseline at 125/130 (96.2%). Its frozen semantic
intervention produced mean target-vs-source log-odds gain +3.53 (95% scenario-
bootstrap interval +2.68 to +4.29), 91.2% positive direction, and a +2.82
advantage over the stronger of random and unrelated controls. Output KL 0.261,
mean delta-RMS ratio 0.0143, basis conditioning, and cosine checks were all
inside the frozen validity domain.

H0R-C nevertheless failed because target-answer top-1 conversion was 4/125
(3.2%), below the preregistered 20% minimum. This was the only failed H0R-C
criterion, but the gate is conjunctive. H0R-D was therefore not run.

The original H0 remains failed. No
`configs/v2/validated_qwen_intervention_protocol.json` exists, no strategic-
reporting experiment is licensed, and none was run. A future causal attempt
requires a new preregistration, model/lens change, or redesigned instrument;
the present causal V2 branch terminates cleanly.
