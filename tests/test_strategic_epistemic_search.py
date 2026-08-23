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
CONFIRMATION = ROOT / "configs/v4/strategic_epistemic_search/report_confirmation.json"
ORDINAL_BINDING = (
    ROOT / "configs/v4/strategic_epistemic_search/ordinal_binding_permutation.json"
)


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_confirmation_protocol_is_held_out_and_minimal() -> None:
    protocol = json.loads(CONFIRMATION.read_text(encoding="utf-8"))["self_report"]
    assert protocol["splits"] == ["validation", "locked"]
    assert protocol["eligible_selected_actions"] == ["C"]
    assert protocol["query_types"] == ["predicted_response"]
    assert set(protocol["access_conditions"]) == {"retrospective", "answer_only"}
    assert protocol["primary_test"] == "two_sided_exact_mcnemar"


def test_arbitrary_response_aliases_preserve_the_target() -> None:
    row = dataset_payload(_config())["rows"][0]
    indexed = report_spec(
        row, "predicted_response", "answer_only", row["expected_action"]
    )
    aliased = report_spec(
        row,
        "predicted_response",
        "answer_only",
        row["expected_action"],
        response_aliases=("Kestrel", "Lumen", "Quartz"),
    )
    assert indexed["correct_value"] == aliased["correct_value"]
    assert indexed["expected_label"] == aliased["expected_label"]
    assert set(aliased["options"].values()) == {"Kestrel", "Lumen", "Quartz"}
    assert "Kestrel=R1" in aliased["messages"][-1]["content"]


def test_ordinal_binding_protocol_is_prospective_and_clustered() -> None:
    config = json.loads(ORDINAL_BINDING.read_text(encoding="utf-8"))
    report = config["self_report"]
    assert config["status"] == "preregistered_before_dataset_freeze_or_model_execution"
    assert config["dataset"]["seed"] != _config()["dataset"]["seed"]
    assert config["dataset"]["ordinal_binding_oa"] == "OA(9,4,3,2)"
    assert report["primary_tests"] == [
        "action_label_to_response_label_lure",
        "action_position_to_response_position_lure",
    ]
    assert report["cluster_unit"] == "base_game_id"
    assert report["test"] == "two_sided_exact_cluster_sign_flip"
    assert report["family_correction"] == "holm_across_two_primary_lures"


def test_behavior_and_report_stages_are_separated_in_remote_entrypoints() -> None:
    source = (ROOT / "modal_strategic_epistemic_search.py").read_text(
        encoding="utf-8"
    )
    behavior_block = source.split("def behavior() -> None:", 1)[1].split(
        "def _confirmation_config", 1
    )[0]
    ordinal_block = source.split("def ordinal_behavior() -> None:", 1)[1].split(
        "def ordinal_report()", 1
    )[0]
    assert "include_self_report=False" in behavior_block
    assert "include_self_report=False" in ordinal_block


def test_ordinal_binding_dataset_is_pairwise_orthogonal() -> None:
    config = json.loads(ORDINAL_BINDING.read_text(encoding="utf-8"))
    payload = dataset_payload(config)
    verify_dataset_payload(payload, config)
    assert len(payload["rows"]) == 360
    group = [
        row
        for row in payload["rows"]
        if row["base_game_id"] == payload["rows"][0]["base_game_id"]
        and row["frame"] == payload["rows"][0]["frame"]
    ]
    assert len(group) == 9
    factors = (
        "action_label_shift",
        "action_order_shift",
        "response_label_shift",
        "response_order_shift",
    )
    for left_index, left in enumerate(factors):
        for right in factors[left_index + 1 :]:
            observed = {
                (row["ordinal_binding"][left], row["ordinal_binding"][right])
                for row in group
            }
            expected = {
                (left_level, right_level)
                for left_level in range(3)
                for right_level in range(3)
            }
            assert observed == expected


def test_ordinal_binding_prompt_obeys_frozen_presentation_orders() -> None:
    config = json.loads(ORDINAL_BINDING.read_text(encoding="utf-8"))
    row = dataset_payload(config)["rows"][0]
    certificate = row["ordinal_binding"]
    action_order = certificate["action_presentation_order"]
    response_order = certificate["response_presentation_order"]
    action_surfaces = [f"{label}=(" for label in "ABC"]
    response_surfaces = [f"R{index + 1}" for index in range(3)]
    assert [row["prompt"].index(action_surfaces[index]) for index in action_order] == sorted(
        row["prompt"].index(action_surfaces[index]) for index in action_order
    )
    header = row["prompt"].split("P(", 1)[1].split(" |", 1)[0]
    assert header.split(",") == [response_surfaces[index] for index in response_order]


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
        common = {
            "decision_prompt": "frozen decision prompt",
            "trajectory": f"reasoning\nFINAL: {row['expected_action']}",
            "matched_trajectory": f"control\nFINAL: {row['expected_action']}",
            "ordinal_trajectory": (
                f"the first signal wins\nFINAL: {row['expected_action']}"
            ),
            "system_prompt": "system",
            "report_labels": tuple(config["self_report"]["candidate_labels"]),
        }
        specs = {
            access: report_spec(
                row,
                query_type,
                access,
                row["expected_action"],
                **common,
            )
            for access in config["self_report"]["access_conditions"]
        }
        retrospective = specs["retrospective"]
        assert all(
            retrospective["expected_label"] == spec["expected_label"]
            for spec in specs.values()
        )
        assert all(retrospective["options"] == spec["options"] for spec in specs.values())
        assert retrospective == report_spec(
            row,
            query_type,
            "retrospective",
            row["expected_action"],
            **common,
        )
        assert set(retrospective["options"]) == {"X", "Y", "Z"}
        assert retrospective["expected_label"] in retrospective["options"]
        assert [message["role"] for message in retrospective["messages"]] == [
            "system",
            "user",
            "assistant",
            "user",
        ]
        assert [message["role"] for message in specs["answer_only"]["messages"]] == [
            "system",
            "user",
            "assistant",
            "user",
        ]
        assert [message["role"] for message in specs["matched_trajectory"]["messages"]] == [
            "system",
            "user",
            "assistant",
            "user",
        ]
        assert [message["role"] for message in specs["ordinal_trajectory"]["messages"]] == [
            "system",
            "user",
            "assistant",
            "user",
        ]
        reconstruction = specs["reconstruction"]
        assert [message["role"] for message in reconstruction["messages"]] == [
            "system",
            "user",
        ]
        assert "hidden computation" in reconstruction["messages"][-1]["content"]


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
