# Tests

These tests check the reusable local code without requiring a GPU or a model
download. They are guardrails for the data, statistics, intervention math,
and diagnostics used by the larger runs.

Run them with:

```bash
uv run pytest
```

| Test file | What it protects |
|---|---|
| [`test_dataset.py`](test_dataset.py) | Deterministic prompt construction and balanced conditions. |
| [`test_analysis.py`](test_analysis.py) | Paired estimates and summary calculations. |
| [`test_interventions.py`](test_interventions.py) | Direction, ablation, coordinate-swap, and hook behavior. |
| [`test_h0r_diagnostics.py`](test_h0r_diagnostics.py) | H0R diagnostic measurements and validity checks. |
| [`test_budget.py`](test_budget.py) | Compute-budget bookkeeping. |
| [`test_strategic_trajectories.py`](test_strategic_trajectories.py) and [`test_workspace_atlas.py`](test_workspace_atlas.py) | V2 strategic corpus construction and solver checks. |
| [`test_jspace_interventions.py`](test_jspace_interventions.py) | V3 intervention dataset and control invariants. |
| [`test_strategic_epistemic_search.py`](test_strategic_epistemic_search.py) and [`test_v4_ordinal_binding_*.py`](.) | V4 game construction and report diagnostics. |
| [`test_revealed_belief_games.py`](test_revealed_belief_games.py), [`test_semantic_override_games.py`](test_semantic_override_games.py), [`test_semantic_capture_localization.py`](test_semantic_capture_localization.py), and [`test_inverse_evidence_games.py`](test_inverse_evidence_games.py) | V5 game construction, localization, and inverse-evidence checks. |
| [`test_mechanistic_decomposition_games.py`](test_mechanistic_decomposition_games.py) | V5-RBG-5 focused corpus, behavioral gate, natural donors, and patch-freeze logic. |
| [`test_mechanistic_decomposition_b.py`](test_mechanistic_decomposition_b.py) | V5-RBG-5B fresh corpus, non-inferiority gate, exact BF16 storage, and geometry fixture. |
| [`test_full_action_trajectory.py`](test_full_action_trajectory.py) | RBG-6 source hash, report-order, and transcript invariants. |

Passing these tests does not prove that an intervention works on a model. It
only means the local reusable pieces behave as expected.

[Back to the project README](../README.md)
