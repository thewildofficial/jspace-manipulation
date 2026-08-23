from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

from jspace_policy.strategic_epistemic_search import (
    REPORT_QUERY_TYPES,
    GameState,
    canonical_sha256,
    dataset_payload,
    report_spec,
    verify_dataset_payload,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v4/strategic_epistemic_search/experiment.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_game_state_exact_values_and_pathway() -> None:
    state = GameState(
        policy=(
            (Fraction("0.80"), Fraction("0.15"), Fraction("0.05")),
            (Fraction("0.05"), Fraction("0.80"), Fraction("0.15")),
            (Fraction("0.15"), Fraction("0.05"), Fraction("0.80")),
        ),
        payoffs=(10, 2, -4),
        costs=(1, 0, 0),
    )
    assert state.values == (Fraction("7.1"), Fraction("1.5"), Fraction("-1.6"))
    assert state.ranking == (0, 1, 2)
    assert state.margin == Fraction("5.6")
    assert state.decisive_response == 0


def test_dataset_is_deterministic_and_valid() -> None:
    config = _config()
    left = dataset_payload(config)
    right = dataset_payload(config)
    assert canonical_sha256(left) == canonical_sha256(right)
    verify_dataset_payload(left, config)


def test_dataset_factorial_size_and_pair_types() -> None:
    config = _config()
    payload = dataset_payload(config)
    assert len(payload["rows"]) == 240
    assert {row["pair_type"] for row in payload["rows"]} == set(
        config["dataset"]["pair_types"]
    )
    assert {row["frame"] for row in payload["rows"]} == {"strategic", "nonagentic"}


def test_tampering_is_detected() -> None:
    config = _config()
    payload = dataset_payload(config)
    payload["rows"][0]["expected_action"] = "Z"
    try:
        verify_dataset_payload(payload, config)
    except ValueError as error:
        assert "hash" in str(error)
    else:
        raise AssertionError("tampered dataset was accepted")


def test_report_factorial_is_deterministic_and_forced_choice() -> None:
    config = _config()
    row = dataset_payload(config)["rows"][0]
    for query_type in REPORT_QUERY_TYPES:
        retrospective = report_spec(
            row, query_type, "retrospective", row["expected_action"]
        )
        reconstruction = report_spec(
            row, query_type, "reconstruction", row["expected_action"]
        )
        assert retrospective["expected_label"] == reconstruction["expected_label"]
        assert retrospective["options"] == reconstruction["options"]
        assert retrospective == report_spec(
            row, query_type, "retrospective", row["expected_action"]
        )
        assert set(retrospective["options"]) == {"A", "B", "C"}
        assert retrospective["expected_label"] in retrospective["options"]
        assert [message["role"] for message in retrospective["messages"]] == [
            "user",
            "assistant",
            "user",
        ]
        assert [message["role"] for message in reconstruction["messages"]] == ["user"]
        assert "hidden computation" in reconstruction["messages"][0]["content"]


def test_report_target_matches_frozen_game_certificates() -> None:
    config = _config()
    for row in dataset_payload(config)["rows"][:20]:
        decisive = report_spec(
            row, "decisive_response", "retrospective", row["expected_action"]
        )
        predicted = report_spec(
            row, "predicted_response", "retrospective", row["expected_action"]
        )
        margin = report_spec(
            row, "decision_margin", "retrospective", row["expected_action"]
        )
        assert decisive["correct_value"] == str(row["decisive_response"])
        selected = int(row["winner"])
        expected_prediction = max(
            range(3), key=lambda index: Fraction(row["policy"][selected][index])
        )
        assert predicted["correct_value"] == str(expected_prediction)
        assert margin["correct_value"] == row["margin"]
