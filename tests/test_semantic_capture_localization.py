from __future__ import annotations

import json
from pathlib import Path

from jspace_policy.semantic_capture_localization import (
    dataset_payload,
    rehearsed_action_messages,
    verify_dataset_payload,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v5/semantic_localization/experiment.json"
MANIFEST = ROOT / "configs/v5/semantic_localization/dataset_manifest.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_factorial_is_deterministic_and_valid() -> None:
    payload = dataset_payload(_config())
    verify_dataset_payload(payload, _config())
    assert payload == dataset_payload(_config())
    assert len(payload["rows"]) == 384
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert payload["content_sha256"] == manifest["expected_content_sha256"]


def test_surface_variants_preserve_causal_game() -> None:
    rows = dataset_payload(_config())["rows"]
    grouped: dict[tuple[str, ...], list[dict]] = {}
    for row in rows:
        key = (row["base_game_id"], row["frame"], row["incentive"])
        grouped.setdefault(key, []).append(row)
    for variants in grouped.values():
        assert {row["surface_kind"] for row in variants} == {
            "assertion",
            "quoted_string",
            "labeled_button",
            "opaque_token",
        }
        assert len({row["expected_action"] for row in variants}) == 1
        assert len({tuple(row["response_per_action"].values()) for row in variants}) == 1


def test_rehearsal_supplies_both_correct_consequences() -> None:
    for row in dataset_payload(_config())["rows"]:
        messages = rehearsed_action_messages(row, "system")
        assistant = [item["content"] for item in messages if item["role"] == "assistant"]
        assert len(assistant) == 2
        assert set(assistant) <= {"X", "Y"}
        assert messages[-1]["content"].endswith("Answer:")
