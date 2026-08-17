# V2 Stage 1 results

Stage 1 tested prospective latent state–report dissociation under neutral
four-state reporting policies. The experiment is observational; it does not
repair the failed H0/H0R causal gates.

## Locked result

Both locked behavioral gates passed:

- Stage 1A explicit state: 240/240 exact reports.
- Stage 1B inferred state: 237/240 exact reports (98.75%); every cell and family
  remained above its frozen threshold.

The conjunctive J-space criterion failed:

| Substage | Transformed band K (95% scenario-bootstrap CI) | Positive families | State probe |
|---|---:|---:|---:|
| 1A | -0.108 [-0.142, -0.073] | 0/3 | 1.000 [1.000, 1.000] |
| 1B | -0.017 [-0.043, 0.008] | 2/3 | 0.992 [0.979, 1.000] |

State information was strongly decodable by the independent discovery-trained
residual probes, including Stage 1B cross-policy transfer (70.0% truthful-to-
transformed and 49.2% transformed-to-truth versus 25% chance). It was not
reliably surfaced by the pinned vocabulary-grounded J-space score. This matches
the preregistered interpretation for a residual-positive/J-space-negative
outcome, so Stage 2 is not licensed.

## Artifact map

- `raw/behavior_dev.json` and `raw/behavior_locked.json`: valid behavior gates.
- `raw/mechanistic_dev.json` and `raw/mechanistic_locked.json`: immutable
  prompt-level J-space, logit-lens, output, and frozen-probe results.
- `raw/behavior_dev_v*_failed_gate.json`: retained invalid behavior-only task
  iterations; no mechanistic output was opened for them.
- `summaries/`: deterministic behavior, primary, family, trajectory, and probe
  tables.
- `figures/stage1_primary_results.png`: primary locked visualization.
- `cost_ledger.jsonl`: measured Modal run-cost ledger.
- `run_manifest.json`: pinned model/lens/software/seeds plus hashes and run IDs
  for all four valid raw result files.

Regenerate every final table, figure, and the Markdown report with:

```bash
uv run python scripts/analyze_stage1.py --phase locked
```

The full preregistration, machine analysis definitions, locked freeze, final
report, and append-only decisions are under `docs/v2/`.
