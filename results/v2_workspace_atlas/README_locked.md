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

## Exploratory strategy decoding

Best layer is selected post hoc and is descriptive only.

```text
      game representation  layer  balanced_accuracy
 chameleon         jspace     12              1.000
 chameleon         output     -1              0.750
 chameleon       residual     18              1.000
cheap_talk         jspace     55              0.500
cheap_talk         output     -1              0.500
cheap_talk       residual     19              0.833
disclosure         jspace      8              0.500
disclosure         output     -1              1.000
disclosure       residual     42              1.000
inspection         jspace     62              0.500
inspection         output     -1              1.000
inspection       residual     59              1.000
      kuhn         jspace     59              0.375
      kuhn         output     -1              0.312
      kuhn       residual     51              0.625
 signaling         jspace     50              0.333
 signaling         output     -1              0.333
 signaling       residual     43              0.778
```

## Output commitment depth

```text
      game  median_commitment_layer  measured
 chameleon                   50.000         9
cheap_talk                   47.000         9
disclosure                   50.000        12
inspection                   48.000        12
      kuhn                   60.000         5
 signaling                   60.000        11
```

## Artifact map

- `raw/`: immutable behavior and mechanistic returns.
- `summaries/`: behavior, probes, token inventory, echo rates, and commitment tables.
- `atlas/`: raw top-token inventory suitable for qualitative inspection.
- `figures/`: deterministic behavioral and probe plots.

All mechanistic patterns in the open phase are discovery findings until a replication freeze is committed.

## Frozen replication endpoints

```text
                         endpoint           unit  primary_value  secondary_value  tertiary_value  pass                                                 detail
H1_generic_optimization_workspace           kuhn          1.000           27.500           1.000  True presence; median best rank; fraction before commitment
H1_generic_optimization_workspace      signaling          1.000           11.000           1.000  True presence; median best rank; fraction before commitment
         H2_strategy_routing_game     inspection          1.000            0.500           1.000 False            residual; J-space; output balanced accuracy
         H2_strategy_routing_game           kuhn          0.375            0.250           0.312  True            residual; J-space; output balanced accuracy
         H2_strategy_routing_game     cheap_talk          0.833            0.500           0.500  True            residual; J-space; output balanced accuracy
         H2_strategy_routing_game      signaling          0.389            0.333           0.333  True            residual; J-space; output balanced accuracy
         H2_strategy_routing_game     disclosure          1.000            0.500           1.000 False            residual; J-space; output balanced accuracy
              H2_strategy_routing five_game_mean          0.719            0.090           3.000 False            mean residual BA; mean advantage; games won
     H3_same_action_kuhn_strategy         kuhn_A          0.333            0.333           0.417 False         residual; J-space; output BA; advantage=-0.083
        H4_late_action_commitment           kuhn         63.000            0.583          12.000  True      censored median layer; uncommitted fraction; rows
        H4_late_action_commitment      signaling         60.000            0.083          12.000  True      censored median layer; uncommitted fraction; rows
```
