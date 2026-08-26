# Stage 1E.4a — H0R-C corpus eligibility check

## The question

After freezing the H0R candidate, could a fresh set of argument-substitution
controls support a fair prospective test?

The answer was no. The corpus itself was too difficult for the unmodified
model, so the runner stopped before opening the intervention.

## What happened

The 102-trial corpus reached **60.8% baseline accuracy**, below the locked 80%
minimum:

| Family | Baseline accuracy |
|---|---:|
| Unseen countries | 86.7% |
| Greek sequences | 50.0% |
| Chemical elements | 50.0% |
| Weekdays | 50.0% |

Because the model often missed the ordinary answer, we could not fairly ask
whether an intervention made it produce a counterfactual answer. The run
recorded `intervention_opened: false`, and there is no intervention output.

## Why this is not a causal result

This is an invalid positive-control corpus, not evidence for or against the
frozen protocol. The failure is still important because it shows the value of
checking baseline competence before interpreting an intervention.

The preregistration allowed one simpler replacement corpus without changing
the intervention, causal thresholds, or validity limits. That replacement is
[`../v2_h0r_argument_validation_v2/README.md`](../v2_h0r_argument_validation_v2/README.md).

The corpus definitions are in [`../../src/jspace_policy/h0r.py`](../../src/jspace_policy/h0r.py)
and the prospective runner is [`../../modal_h0r.py`](../../modal_h0r.py).

[H0R diagnosis](../v2_h0r_diagnostic/README.md) · [Results map](../README.md)
