# V2-E1 Strategic Workspace Atlas

Phase: **locked**.

This is an observational exploratory result. It does not reopen the failed causal gate.

## Behavioral gate

- Gate: **pass**
- Formatting compliance: 100.0%
- Exact-game optimal accuracy: 85.0%
- Minimum exact-game family accuracy: 66.7%

```text
             split       game  n  accuracy  legal_choice_accuracy  formatting_compliance  mean_regret  mean_output_entropy
locked_replication  chameleon 12     0.250                  0.250                  1.000        0.225                0.742
locked_replication cheap_talk 12     0.833                  0.833                  1.000        0.208                0.482
locked_replication disclosure 12     1.000                  1.000                  1.000        0.000                0.032
locked_replication inspection 12     1.000                  1.000                  1.000        0.000                0.009
locked_replication       kuhn 12     0.667                  0.667                  1.000        0.337                0.814
locked_replication  signaling 12     0.750                  0.750                  1.000        0.068                0.884
```

## Artifact map

- `raw/`: immutable behavior and mechanistic returns.
- `summaries/`: behavior, probes, token inventory, echo rates, and commitment tables.
- `atlas/`: raw top-token inventory suitable for qualitative inspection.
- `figures/`: deterministic behavioral and probe plots.

All mechanistic patterns in the open phase are discovery findings until a replication freeze is committed.
