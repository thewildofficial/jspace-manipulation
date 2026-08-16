# H0R-C prospective argument validation

Status: **failed**.

This run used the candidate frozen at `6c42af6` without changing its layers,
mask, operation, alpha, normalization, validity limits, or pass thresholds.

The replacement corpus passed its behavioral gate: 125/130 trials (96.2%) had
the natural source answer top-1. The frozen semantic intervention then produced
mean target-vs-source log-odds gain +3.53 (scenario-bootstrap 95% interval +2.68
to +4.29), 91.2% positive gain, and a +2.82 advantage over the stronger of the
random and unrelated controls. Mean output KL was 0.261 nats, mean delta-RMS
ratio was 0.0143, and the conditioning/cosine validity checks passed.

However, only 4/125 eligible trials (3.2%) made the counterfactual target answer
top-1, below the preregistered 20% requirement. H0R-C therefore fails despite
the strong directional effect. H0R-D is not licensed and was not run. No
`validated_qwen_intervention_protocol.json` is created.
