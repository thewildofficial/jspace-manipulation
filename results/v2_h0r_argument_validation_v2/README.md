# Stage 1E.4b — H0R-C prospective validation

## The question

Did the frozen H0R candidate reliably move the model to a new answer on an
unseen argument-control corpus?

This was the important fresh test. The replacement corpus and the candidate
protocol were fixed before this run’s intervention results were opened.

## What happened

The replacement corpus passed the baseline check: **125/130 trials (96.2%)**
had the natural source answer top-1. The intervention then showed several
strong signs of directional and selective movement:

- mean target-vs-source score gain: **+3.53**;
- 95% scenario-bootstrap interval: **+2.68 to +4.29**;
- positive movement: **91.2%** of eligible trials;
- semantic advantage over the stronger matched control: **+2.82**;
- mean output KL: **0.261 nats**;
- mean residual-change ratio: **0.0143**.

But only **4/125 eligible trials (3.2%)** made the target answer top-1. The
locked requirement was 20%. Because the gate was conjunctive, H0R-C **failed**.

## The important distinction

The intervention often pushed the answer score in the requested direction. It
rarely pushed far enough for the requested answer to become the model’s first
choice. That is evidence of partial causal access, not a reliable
counterfactual answer-control instrument.

This is also why the result is not “almost a pass.” The target-top-1 criterion
was included precisely to distinguish a score shift from a dependable answer
change.

## What was not run

H0R-D, the computed-intermediate control, was not licensed. The strategic
reporting task was not run. No validated protocol file was created.

## Evidence files

- [`raw/baseline.jsonl`](raw/baseline.jsonl): fresh baseline rows.
- [`raw/interventions.jsonl`](raw/interventions.jsonl): semantic and control intervention rows.
- [`primary_outcomes.csv`](primary_outcomes.csv): primary outcome table.
- [`summary.json`](summary.json): gate and validity summary.
- [`run_manifest.json`](run_manifest.json): model, code, corpus, and run metadata.
- [`../../docs/v2/h0r-final-report.md`](../../docs/v2/h0r-final-report.md): the final interpretation.

The fresh corpus is defined in [`../../configs/v2/h0r_locked_controls_v2.json`](../../configs/v2/h0r_locked_controls_v2.json)
and [`../../src/jspace_policy/h0r.py`](../../src/jspace_policy/h0r.py); the
runner is [`../../modal_h0r.py`](../../modal_h0r.py).

[Previous recovery phase: H0R-B diagnosis](../v2_h0r_diagnostic/README.md) · [Eligibility check](../v2_h0r_argument_validation/README.md) · [Results map](../README.md)
