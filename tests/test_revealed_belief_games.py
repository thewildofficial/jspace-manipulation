from __future__ import annotations

import json
from pathlib import Path

from jspace_policy.revealed_belief_games import (
    dataset_payload,
    report_question,
    verify_dataset_payload,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v5/revealed_belief_games/experiment.json"
MANIFEST = ROOT / "configs/v5/revealed_belief_games/dataset_manifest.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_dataset_is_deterministic_and_valid() -> None:
    left = dataset_payload(_config())
    right = dataset_payload(_config())
    assert left == right
    verify_dataset_payload(left, _config())
    assert len(left["rows"]) == 192
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert left["content_sha256"] == manifest["expected_content_sha256"]
    assert len(left["rows"]) == manifest["expected_contexts"]


def test_type_counterfactual_changes_unique_best_response() -> None:
    rows = dataset_payload(_config())["rows"]
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        key = (row["base_game_id"], row["frame"], row["incentive"])
        grouped.setdefault(key, []).append(row)
    assert grouped
    for pair in grouped.values():
        assert {row["receiver_type"] for row in pair} == {"literal", "contrarian"}
        assert len({row["expected_action"] for row in pair}) == 2


def test_opposed_games_induce_false_receiver_response() -> None:
    rows = dataset_payload(_config())["rows"]
    for row in rows:
        if row["incentive"] == "opposed":
            assert row["expected_receiver_response"] != row["true_state"]
        else:
            assert row["expected_receiver_response"] == row["true_state"]


def test_report_is_private_forced_choice() -> None:
    row = dataset_payload(_config())["rows"][0]
    question = report_question(row, row["expected_action"])
    assert "Return only X or Y" in question
    assert "cannot affect any payoff" in question
    assert row["concepts"][0] in question and row["concepts"][1] in question
