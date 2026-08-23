from __future__ import annotations

import json
from pathlib import Path

from jspace_policy.semantic_override_games import dataset_payload, verify_dataset_payload

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v5/semantic_override/experiment.json"
MANIFEST = ROOT / "configs/v5/semantic_override/dataset_manifest.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_factorial_is_deterministic_and_valid() -> None:
    payload = dataset_payload(_config())
    verify_dataset_payload(payload, _config())
    assert payload == dataset_payload(_config())
    assert len(payload["rows"]) == 512
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert payload["content_sha256"] == manifest["expected_content_sha256"]


def test_semantic_pair_preserves_causal_game() -> None:
    rows = dataset_payload(_config())["rows"]
    grouped: dict[tuple[str, ...], list[dict]] = {}
    for row in rows:
        key = (
            row["base_game_id"],
            row["frame"],
            row["incentive"],
            row["receiver_type"],
            row["policy_access"],
        )
        grouped.setdefault(key, []).append(row)
    for pair in grouped.values():
        assert {row["message_semantics"] for row in pair} == {
            "claims",
            "opaque_tokens",
        }
        assert len({row["expected_action"] for row in pair}) == 1
        assert len({tuple(row["response_per_action"].values()) for row in pair}) == 1


def test_explicit_mapping_contains_both_consequences() -> None:
    rows = dataset_payload(_config())["rows"]
    for row in rows:
        if row["policy_access"] == "explicit":
            assert "Verified current policy" in row["prompt"]
            assert all(value in row["prompt"] for value in row["response_per_action"].values())
