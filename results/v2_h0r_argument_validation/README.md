# H0R-C first argument-control baseline

Status: **behavioral gate failed; intervention unopened**.

The frozen 102-trial argument control achieved 60.8% baseline accuracy, below
the preregistered 80% minimum. Accuracy by family was 86.7% for unseen
countries and 50% each for Greek sequences, chemical elements, and weekdays.

The runner therefore stopped before intervention inference. The run manifest
records `intervention_opened: false`, and no `raw/interventions.jsonl` exists.
This is an invalid positive-control corpus, not evidence for or against the
frozen causal protocol. Per the preregistration, a simpler replacement control
may be frozen and baseline-checked without changing the protocol or causal
thresholds.
