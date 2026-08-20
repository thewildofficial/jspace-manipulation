# V2-E1 Strategic Workspace Atlas

Phase: **open**.

This is an observational exploratory result. It does not reopen the failed causal gate.

## Behavioral gate

- Gate: **fail**
- Formatting compliance: 100.0%
- Exact-game optimal accuracy: 70.0%
- Minimum exact-game family accuracy: 41.7%

```text
     split       game  n  accuracy  legal_choice_accuracy  formatting_compliance  mean_regret  mean_output_entropy
 discovery  chameleon 12     0.500                  0.500                  1.000        0.163                0.863
 discovery cheap_talk 12     0.917                  0.917                  1.000        0.250                0.198
 discovery disclosure 12     0.750                  0.750                  1.000        1.125                0.889
 discovery inspection 12     0.417                  0.417                  1.000        1.967                0.697
 discovery       kuhn 12     0.417                  0.417                  1.000        0.160                0.545
 discovery  signaling 12     0.833                  0.833                  1.000        0.079                1.015
validation  chameleon 12     0.750                  0.750                  1.000        0.062                0.929
validation cheap_talk 12     0.833                  0.833                  1.000        0.583                0.854
validation disclosure 12     0.750                  0.750                  1.000        1.125                1.013
validation inspection 12     0.917                  0.917                  1.000        0.017                0.704
validation       kuhn 12     0.417                  0.417                  1.000        0.160                0.568
validation  signaling 12     0.750                  0.750                  1.000        0.135                1.057
```

## Artifact map

- `raw/`: immutable behavior and mechanistic returns.
- `summaries/`: behavior, probes, token inventory, echo rates, and commitment tables.
- `atlas/`: raw top-token inventory suitable for qualitative inspection.
- `figures/`: deterministic behavioral and probe plots.

All mechanistic patterns in the open phase are discovery findings until a replication freeze is committed.
