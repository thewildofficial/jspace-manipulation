from __future__ import annotations

import re
from collections import Counter

import pytest

from jspace_policy.stage1 import (
    BANNED_PRIMARY_WORDS,
    STAGE1A_FAMILIES,
    STAGE1B_FAMILIES,
    dataset_payload,
    generate_stage1_dataset,
    verify_dataset_payload,
)
from jspace_policy.stage1_analysis import (
    macro_ovr_auc,
    paired_policy_effect,
    scenario_bootstrap,
)


@pytest.fixture
def config() -> dict:
    return {"dataset_seed": 421337, "base_scenarios_per_family": 10}


def test_dataset_is_complete_balanced_and_deterministic(config: dict) -> None:
    first = generate_stage1_dataset(config)
    second = generate_stage1_dataset(config)
    assert first == second
    assert len(first) == 1920
    assert len(STAGE1A_FAMILIES) == len(STAGE1B_FAMILIES) == 12
    assert len({row.base_scenario_id for row in first}) == 240
    grouped = Counter(
        (row.base_scenario_id, row.world_state_id, row.policy_id) for row in first
    )
    assert set(grouped.values()) == {1}
    assert Counter(row.split for row in first) == {
        "discovery": 960,
        "validation": 480,
        "locked_test": 480,
    }


def test_transformation_is_deranged_and_reports_are_balanced(config: dict) -> None:
    rows = generate_stage1_dataset(config)
    transformed = [row for row in rows if row.policy_id == "M"]
    assert all(row.true_state != row.expected_report for row in transformed)
    for substage in ("1A", "1B"):
        part = [row for row in rows if row.substage == substage]
        counts = Counter((row.policy_id, row.expected_report) for row in part)
        by_policy = Counter(row.policy_id for row in part)
        labels = {row.expected_report for row in part}
        for policy, total in by_policy.items():
            assert {count for (current, _), count in counts.items() if current == policy} == {
                total // len(labels)
            }


def test_primary_prompts_avoid_banned_vocabulary(config: dict) -> None:
    for row in generate_stage1_dataset(config):
        lowered = row.prompt.lower()
        assert not any(
            re.search(rf"\b{re.escape(word)}\b", lowered) for word in BANNED_PRIMARY_WORDS
        )


def test_dataset_hash_detects_mutation(config: dict) -> None:
    payload = dataset_payload(config)
    verify_dataset_payload(payload)
    payload["rows"][0]["prompt"] += " changed"
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_dataset_payload(payload)


def test_paired_policy_effect_sign() -> None:
    rows = []
    for scenario in ("a", "b"):
        for state in range(4):
            rows.extend(
                [
                    {
                        "base_scenario_id": scenario,
                        "world_state_id": state,
                        "policy_id": "T",
                        "score": 1.0,
                    },
                    {
                        "base_scenario_id": scenario,
                        "world_state_id": state,
                        "policy_id": "M",
                        "score": 3.0,
                    },
                ]
            )
    assert paired_policy_effect(rows, "score") == pytest.approx(2.0)


def test_scenario_bootstrap_keeps_factorial_groups() -> None:
    rows = [
        {"base_scenario_id": scenario, "value": value}
        for scenario, value in (("a", 1.0), ("a", 1.0), ("b", 3.0), ("b", 3.0))
    ]
    point, low, high = scenario_bootstrap(
        rows,
        lambda sample: sum(row["value"] for row in sample) / len(sample),
        draws=200,
        seed=5,
    )
    assert point == pytest.approx(2.0)
    assert low >= 1.0
    assert high <= 3.0


def test_macro_auc_is_one_for_perfect_predictions() -> None:
    targets = [0, 1, 2, 3] * 3
    probabilities = [
        [0.97 if index == target else 0.01 for index in range(4)] for target in targets
    ]
    assert macro_ovr_auc(targets, probabilities) == pytest.approx(1.0)
