from __future__ import annotations

import pytest

from jspace_policy.sprint_analysis import (
    grouped_paired_analysis,
    incident_gate,
    paired_base_effects,
    power_simulation,
    primary_gate,
)


def _rows(effects: list[float], *, split: str = "discovery") -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, effect in enumerate(effects):
        # The extra treatment row exercises within-base averaging.  It should
        # not make this base count twice in the primary estimand.
        rows.extend(
            [
                {
                    "base_id": f"b{index}",
                    "split": split,
                    "arm": "self_report",
                    "correct": effect,
                },
                {
                    "base_id": f"b{index}",
                    "split": split,
                    "arm": "matched_control",
                    "correct": 0.0,
                },
                {
                    "base_id": f"b{index}",
                    "split": split,
                    "arm": "self_report",
                    "correct": effect,
                },
            ]
        )
    return rows


def test_paired_effects_weight_bases_equally() -> None:
    rows = _rows([1.0, 0.0])
    result = paired_base_effects(rows)
    assert result["effect"].tolist() == [1.0, 0.0]
    summary = grouped_paired_analysis(rows, bootstrap_resamples=100, seed=10)
    assert summary["effect"] == pytest.approx(0.5)
    assert summary["n_bases"] == 2


def test_exact_sign_flip_is_seed_independent_for_small_n() -> None:
    rows = _rows([1.0] * 5)
    left = grouped_paired_analysis(rows, bootstrap_resamples=100, seed=1)
    right = grouped_paired_analysis(rows, bootstrap_resamples=100, seed=987)
    assert left["p_value"] == pytest.approx(1 / 32)
    assert right["p_value"] == left["p_value"]


def test_primary_gate_is_unconditional_and_locked_never_selects() -> None:
    rows = _rows([1.0] * 5, split="locked")
    result = primary_gate(rows, split="locked", bootstrap_resamples=100)
    assert result["gate_pass"]
    assert result["selection_allowed"] is False
    assert result["confirmatory"] is False
    assert result["status"] == "descriptive_pilot"


def test_mixed_unspecified_split_cannot_authorize_selection() -> None:
    rows = _rows([1.0] * 5, split="discovery") + _rows([1.0] * 5, split="locked")
    result = primary_gate(rows, bootstrap_resamples=100)
    assert result["gate_pass"]
    assert result["selection_allowed"] is False


def test_primary_gate_fails_closed_for_incomplete_support() -> None:
    rows = _rows([1.0], split="discovery")
    rows = [row for row in rows if row["arm"] != "matched_control"]
    result = primary_gate(rows, split="discovery", bootstrap_resamples=100)
    assert result["gate_pass"] is False
    assert result["decision"] == "insufficient_support"
    assert result["selection_allowed"] is True


def test_power_simulation_is_seeded_and_cpu_only() -> None:
    left = power_simulation(n_bases=12, simulations=200, seed=4)
    right = power_simulation(n_bases=12, simulations=200, seed=4)
    assert left == right
    assert 0.0 <= left["power"] <= 1.0


def test_incident_gate_requires_competence_and_bounds_violations() -> None:
    rows: list[dict[str, object]] = []
    for index in range(8):
        rows.extend(
            [
                {"base_id": index, "split": "pilot", "arm": "competence", "correct": 1},
                {
                    "base_id": index,
                    "split": "pilot",
                    "arm": "violation",
                    "correct": int(index < 2),
                },
            ]
        )
    result = incident_gate(rows, split="pilot", bootstrap_resamples=100)
    assert result["gate_pass"]
    assert result["competence"] == pytest.approx(1.0)
    assert result["n_violating_bases"] == 2
    assert result["status"] == "descriptive_pilot"
    assert result["confirmatory"] is False


def test_incident_gate_does_not_treat_sparse_support_as_pass() -> None:
    rows = [
        {"base_id": "only", "split": "pilot", "arm": "competence", "correct": 1},
        {"base_id": "only", "split": "pilot", "arm": "violation", "correct": 0},
    ]
    result = incident_gate(rows, split="pilot", bootstrap_resamples=100)
    assert result["gate_pass"] is False
    assert result["decision"] == "stop_insufficient_violations"
    assert result["violation_upper"] == pytest.approx(0.95)


def test_zero_incidents_report_finite_cluster_upper_bound_and_stop() -> None:
    rows: list[dict[str, object]] = []
    for index in range(8):
        rows.extend(
            [
                {"base_id": index, "split": "pilot", "arm": "competence", "correct": 1},
                {"base_id": index, "split": "pilot", "arm": "violation", "correct": 0},
            ]
        )
    result = incident_gate(rows, split="pilot", bootstrap_resamples=100)
    assert result["gate_pass"] is False
    assert result["decision"] == "stop_insufficient_violations"
    assert result["violation_upper_bound"] == pytest.approx(1 - 0.05 ** (1 / 8))
