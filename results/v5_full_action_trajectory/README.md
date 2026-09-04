# V5-RBG-6 — endogenous full action trajectory

This result answers [issue #11](https://github.com/thewildofficial/when-words-override-consequences/issues/11): does a model that reports both option consequences, sees its own report tokens in context, and then chooses behave differently from a direct action query?

The result is a strong rescue, not persistence. In the strategic opposed assertion cell with redundant history and prose mapping, self-generated reports were followed by correct actions in 46/48 trajectories; the “both reports correct, action wrong” conjunction occurred 0/48 times. Direct action accuracy on the same immutable RBG-4 contexts was 37.5%, while the full trajectory reached 95.83%.

The experiment also ran oracle-correct and swapped-report replays. Oracle replay was 100% accurate; swapped replay selected the action implied by the inverted reports in 89.58% of cases. Report order was exactly balanced, and the matched opaque-token control stayed at 100%.

See [`FINDINGS.md`](FINDINGS.md) for the frozen decision, cell table, provenance, hashes, and claim boundary.

## Reproduce offline

```bash
uv run --extra dev pytest -q
uv run python scripts/analyze_v5_full_action_trajectory.py \
  results/v5_full_action_trajectory/raw/behavior_v1.json \
  --config configs/v5/full_action_trajectory/experiment.json \
  --output /tmp/v5-rbg6-analysis.json
```
