from __future__ import annotations

import json
from pathlib import Path

from jspace_policy.inverse_evidence_games import dataset_payload, verify_dataset_payload

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v5/inverse_evidence/experiment.json"
MANIFEST = ROOT / "configs/v5/inverse_evidence/dataset_manifest.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_factorial_is_deterministic_and_valid() -> None:
    payload = dataset_payload(_config())
    verify_dataset_payload(payload, _config())
    assert payload == dataset_payload(_config())
    assert len(payload["rows"]) == 768
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert payload["content_sha256"] == manifest["expected_content_sha256"]


def test_history_and_format_preserve_matched_game() -> None:
    rows = dataset_payload(_config())["rows"]
    grouped: dict[tuple[str, ...], list[dict]] = {}
    for row in rows:
        key = (row["base_game_id"], row["frame"], row["incentive"])
        grouped.setdefault(key, []).append(row)
    for variants in grouped.values():
        assert len(variants) == 8
        assert len({row["expected_action"] for row in variants}) == 1
        assert len({tuple(row["response_per_action"].values()) for row in variants}) == 1
        assert {row["history"] for row in variants} == {"none", "redundant"}
        assert {row["mapping_format"] for row in variants} == {"prose", "table"}


def test_redundant_history_contains_four_correct_examples() -> None:
    for row in dataset_payload(_config())["rows"]:
        if row["history"] == "redundant":
            assert row["scenario"].count("Prior ") == 4
        else:
            assert "No prior episodes" in row["scenario"]
