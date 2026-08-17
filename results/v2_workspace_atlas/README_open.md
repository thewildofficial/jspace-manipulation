# V2-E1 Strategic Workspace Atlas

Phase: **open**.

This is an observational exploratory result. It does not reopen the failed causal gate.

## Behavioral gate

- Gate: **pass**
- Formatting compliance: 100.0%
- Exact-game optimal accuracy: 78.3%
- Minimum exact-game family accuracy: 62.5%

```text
     split       game  n  accuracy  legal_choice_accuracy  formatting_compliance  mean_regret  mean_output_entropy
 discovery  chameleon 12     0.500                  0.500                  1.000        0.163                0.871
 discovery cheap_talk 12     0.833                  0.833                  1.000        0.208                0.335
 discovery disclosure 12     1.000                  1.000                  1.000        0.000                0.059
 discovery inspection 12     1.000                  1.000                  1.000        0.000                0.012
 discovery       kuhn 12     0.667                  0.667                  1.000        0.233                0.627
 discovery  signaling 12     0.917                  0.917                  1.000        0.062                1.018
validation  chameleon 12     0.750                  0.750                  1.000        0.062                0.921
validation cheap_talk 12     0.417                  0.417                  1.000        2.396                0.944
validation disclosure 12     0.833                  0.833                  1.000        0.713                0.739
validation inspection 12     0.833                  0.833                  1.000        0.100                0.702
validation       kuhn 12     0.583                  0.583                  1.000        0.375                0.624
validation  signaling 12     0.750                  0.750                  1.000        0.135                1.058
```

## Artifact map

- `raw/`: immutable behavior and mechanistic returns.
- `summaries/`: behavior, probes, token inventory, echo rates, and commitment tables.
- `atlas/`: raw top-token inventory suitable for qualitative inspection.
- `figures/`: deterministic behavioral and probe plots.

All mechanistic patterns in the open phase are discovery findings until a replication freeze is committed.
