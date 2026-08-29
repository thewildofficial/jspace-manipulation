# The reusable code

This package contains the small pieces used by the experiments. It does not
run a whole study by itself; the Modal entry points and scripts assemble these
pieces into a run.

| File | What it does |
|---|---|
| [`dataset.py`](dataset.py) | Builds deterministic toy and control prompts. |
| [`analysis.py`](analysis.py) | Computes paired effects and bootstrap intervals. |
| [`interventions.py`](interventions.py) | Builds J-lens directions and temporary residual-state edits. |
| [`h0r.py`](h0r.py) | Defines the fresh argument and computed-intermediate controls. |
| [`h0r_diagnostics.py`](h0r_diagnostics.py) | Measures coordinates, conditioning, reconstruction, and intervention validity. |
| [`jspace_interventions.py`](jspace_interventions.py) | Implements the V3-JI1 intervention-characterization helpers. |
| [`strategic_epistemic_search.py`](strategic_epistemic_search.py) | Builds the V4 decision and relational-report tasks. |
| [`strategic_trajectories.py`](strategic_trajectories.py) | Builds and checks the V2-ST-1 trajectory corpus. |
| [`workspace_atlas.py`](workspace_atlas.py) | Builds and solves the V2-SWA-1 workspace games. |
| [`revealed_belief_games.py`](revealed_belief_games.py) | Builds the V5 receiver-policy game controls. |
| [`semantic_override_games.py`](semantic_override_games.py) | Builds the V5-RBG-2 action/report dissociation task. |
| [`semantic_capture_localization.py`](semantic_capture_localization.py) | Builds the V5-RBG-3 localization controls. |
| [`inverse_evidence_games.py`](inverse_evidence_games.py) | Builds the V5-RBG-4 history and representation contrasts. |
| [`mechanistic_decomposition_games.py`](mechanistic_decomposition_games.py) | Builds the fresh V5-RBG-5 discovery/locked natural-interchange corpus. |
| [`budget.py`](budget.py) | Keeps the planned compute budget visible. |
| [`plotting.py`](plotting.py) | Shared plotting helpers. |
| [`__init__.py`](__init__.py) | Package entry point. |

The tests in [`../../tests/README.md`](../../tests/README.md) check the most
important local behavior, including prompt generation and intervention math.

[Back to the project README](../../README.md)
