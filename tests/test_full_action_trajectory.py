from __future__ import annotations

import json
from pathlib import Path

from jspace_policy.full_action_trajectory import (
    dataset_payload,
    expected_reports,
    file_sha256,
    swapped_reports,
    trajectory_messages,
    verify_dataset_payload,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v5/full_action_trajectory/experiment.json"
MANIFEST = ROOT / "configs/v5/full_action_trajectory/dataset_manifest.json"


def _inputs() -> tuple[dict, dict]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source_path = ROOT / config["source"]["path"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    return config, source


def test_source_and_factorial_are_frozen() -> None:
    config, source = _inputs()
    source_path = ROOT / config["source"]["path"]
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert file_sha256(source_path) == manifest["source_file_sha256"]
    payload = dataset_payload(config, source)
    verify_dataset_payload(payload, config)
    assert payload == dataset_payload(config, source)
    assert len(payload["rows"]) == 768
    assert payload["content_sha256"] == manifest["expected_content_sha256"]


def test_each_source_context_has_both_report_orders() -> None:
    config, source = _inputs()
    rows = dataset_payload(config, source)["rows"]
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["source_condition_id"], []).append(row)
    assert len(grouped) == 384
    for variants in grouped.values():
        assert {tuple(row["report_order"]) for row in variants} == {
            ("A", "B"),
            ("B", "A"),
        }


def test_transcripts_preserve_answers_as_assistant_turns() -> None:
    config, source = _inputs()
    for row in dataset_payload(config, source)["rows"]:
        expected = expected_reports(row)
        swapped = swapped_reports(row)
        assert all(expected[action] != swapped[action] for action in ("A", "B"))
        messages = trajectory_messages(row, "system", expected)
        assistant = [item["content"] for item in messages if item["role"] == "assistant"]
        assert assistant == [expected[action] for action in row["report_order"]]
        assert messages[-1]["content"].endswith("Answer:")
