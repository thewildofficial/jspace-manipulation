from __future__ import annotations

import json
from pathlib import Path

from jspace_policy.mechanistic_decomposition import (
    analyze_behavior,
    analyze_locked_patches,
    content_hash_valid,
    normalized_recovery,
    select_patch_site,
)
from jspace_policy.mechanistic_decomposition_games import (
    CONCEPT_PAIRS,
    dataset_payload,
    matched_row,
    verify_dataset_payload,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v5/mechanistic_decomposition/experiment.json"
MANIFEST = ROOT / "configs/v5/mechanistic_decomposition/dataset_manifest.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_focused_dataset_is_deterministic_fresh_and_valid() -> None:
    config = _config()
    payload = dataset_payload(config)
    verify_dataset_payload(payload, config)
    assert payload == dataset_payload(config)
    assert len(payload["rows"]) == 672
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert payload["content_sha256"] == manifest["expected_content_sha256"]
    assert len(CONCEPT_PAIRS) == 48
    discovery = {
        concept
        for row in payload["rows"]
        if row["split"] == "discovery"
        for concept in row["concepts"]
    }
    locked = {
        concept
        for row in payload["rows"]
        if row["split"] == "locked"
        for concept in row["concepts"]
    }
    assert discovery.isdisjoint(locked)


def test_semantic_anchors_and_natural_donors_are_matched() -> None:
    rows = dataset_payload(_config())["rows"]
    recipient = next(
        row
        for row in rows
        if row["surface_kind"] == "assertion"
        and row["history"] == "redundant"
        and row["mapping_format"] == "prose"
        and row["incentive"] == "opposed"
    )
    assert recipient["anchor_char_ends"]["answer"] == len(recipient["prompt"])
    assert all(f"history_{index}" in recipient["anchor_char_ends"] for index in range(1, 5))
    for family in ("table", "opaque"):
        donor = matched_row(rows, recipient, family)
        assert donor["base_game_id"] == recipient["base_game_id"]
        assert donor["frame"] == recipient["frame"]
        assert donor["target_response"] == recipient["target_response"]
        assert donor["expected_action"] == recipient["expected_action"]
        assert donor["response_per_action"] == recipient["response_per_action"]


def test_behavior_gate_and_eligible_population() -> None:
    config = _config()
    dataset = dataset_payload(config)
    rows = []
    for source in dataset["rows"]:
        row = dict(source)
        row["action_correct"] = not (
            row["incentive"] == "opposed"
            and row["surface_kind"] == "assertion"
            and row["mapping_format"] == "prose"
            and row["history"] == "redundant"
        )
        row["selected_action"] = (
            row["expected_action"]
            if row["action_correct"]
            else ("B" if row["expected_action"] == "A" else "A")
        )
        row["option_reports"] = {
            action: {"correct": True, "choice": "X", "expected": "X"}
            for action in ("A", "B")
        }
        rows.append(row)
    payload = {
        "metadata": {"run_id": "synthetic"},
        "rows": rows,
    }
    result = analyze_behavior(payload, config)
    assert result["gate_pass"]
    assert result["prose_assertion_history_harm"] == 1.0
    assert result["table_history_harm"] == 0.0
    assert result["locked_eligible_recipients"] == 48


def test_patch_selection_is_deterministic_and_hashed() -> None:
    config = _config()
    rows = []
    for donor in ("table", "opaque"):
        for layer in (36, 37):
            for anchor in ("history_end", "mapping_end"):
                for base in range(3):
                    rows.append(
                        {
                            "donor_family": donor,
                            "layer": layer,
                            "anchor": anchor,
                            "base_game_id": f"m{base:03d}",
                            "margin_change": 1.0,
                        }
                    )
    artifact = select_patch_site(rows, config, "dataset-hash")
    assert content_hash_valid(artifact)
    assert artifact["selected"]["donor_family"] == "table"
    assert artifact["selected"]["layer"] == 36
    assert artifact["selected"]["anchor"] == "history_end"
    assert normalized_recovery(-1.0, 1.0, 0.0, 0.1) == 0.5
    assert normalized_recovery(0.0, 0.05, 1.0, 0.1) is None


def test_locked_patch_analysis_enforces_repair_and_non_damage() -> None:
    primary = [
        {
            "base_game_id": f"m{index:03d}",
            "margin_change": 1.0,
            "repaired": index < 3,
            "normalized_recovery": 0.5,
        }
        for index in range(12)
    ]
    controls = [
        {
            "control": "identity",
            "base_game_id": f"m{index:03d}",
            "margin_change": 0.0,
            "choice_changed": False,
        }
        for index in range(12)
    ]
    for category in ("aligned", "table", "opaque"):
        controls.extend(
            {
                "control": f"non_damage_{category}",
                "base_game_id": f"m{index:03d}",
                "margin_change": 0.0,
                "correct_before": True,
                "correct_after": True,
            }
            for index in range(12)
        )
    controls.extend(
        {
            "control": "reverse_prose_into_success",
            "base_game_id": f"m{index:03d}",
            "margin_change": -1.0,
            "correct_after": True,
        }
        for index in range(12)
    )
    controls.extend(
        {
            "control": "opposite_target_same_base",
            "base_game_id": f"m{index:03d}",
            "margin_change": -1.0,
            "repaired": False,
        }
        for index in range(12)
    )
    reports = [
        {
            "base_game_id": f"m{index:03d}",
            "correct_before": True,
            "correct_after": True,
        }
        for index in range(24)
    ]
    result = analyze_locked_patches(
        {"primary": primary, "controls": controls, "reports": reports}, _config()
    )
    assert result["primary"]["passed"]
    assert result["controls"]["passed"]
    assert result["selective_causal_transport_passed"]
