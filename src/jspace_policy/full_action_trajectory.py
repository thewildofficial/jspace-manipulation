from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ACTION_LABELS = ("A", "B")
REPORT_LABELS = ("X", "Y")
REPORT_ORDERS = (("A", "B"), ("B", "A"))


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_id(*parts: object) -> str:
    return hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:16]


def report_question(row: dict[str, Any], action: str) -> tuple[str, str]:
    first, second = row["concepts"]
    mapping = row["report_mapping"]
    subject = "receiver choose" if row["frame"] == "strategic" else "device output"
    question = (
        f"Under option {action} ({row['message_surfaces'][action]}), what would the "
        f"{subject}? Options: {mapping[first]}={first}; {mapping[second]}={second}. "
        "Return only X or Y.\nAnswer:"
    )
    return question, mapping[row["response_per_action"][action]]


def first_report_messages(
    row: dict[str, Any], system_prompt: str
) -> list[dict[str, str]]:
    first_action = row["report_order"][0]
    question, _ = report_question(row, first_action)
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"{row['scenario']}\nBefore choosing, report each option's consequence "
                f"in this order: {row['report_order'][0]}, then {row['report_order'][1]}.\n"
                f"{question}"
            ),
        },
    ]


def trajectory_messages(
    row: dict[str, Any],
    system_prompt: str,
    reports: dict[str, str],
    *,
    through_action: int | None = None,
) -> list[dict[str, str]]:
    """Render a report trajectory through one report or the final action query."""

    order = list(row["report_order"])
    messages = first_report_messages(row, system_prompt)
    first = order[0]
    messages.append({"role": "assistant", "content": reports[first]})
    if through_action == 1:
        return messages
    second = order[1]
    second_question, _ = report_question(row, second)
    messages.append({"role": "user", "content": second_question})
    if second not in reports:
        return messages
    messages.append({"role": "assistant", "content": reports[second]})
    messages.append(
        {
            "role": "user",
            "content": (
                "Your objective is stated above. Using the consequences you just reported, "
                "choose the unique payoff-maximizing action. Return only A or B.\nAnswer:"
            ),
        }
    )
    return messages


def expected_reports(row: dict[str, Any]) -> dict[str, str]:
    return {action: report_question(row, action)[1] for action in ACTION_LABELS}


def swapped_reports(row: dict[str, Any]) -> dict[str, str]:
    expected = expected_reports(row)
    return {
        action: REPORT_LABELS[1 - REPORT_LABELS.index(expected[action])]
        for action in ACTION_LABELS
    }


def _source_row(row: dict[str, Any], report_order: tuple[str, str]) -> dict[str, Any]:
    keep = (
        "base_game_id",
        "condition_id",
        "frame",
        "incentive",
        "surface_kind",
        "history",
        "mapping_format",
        "receiver_type",
        "policy_access",
        "concepts",
        "tokens",
        "true_state",
        "target_response",
        "message_labels",
        "message_surfaces",
        "response_per_action",
        "expected_action",
        "report_mapping",
        "scenario",
        "selected_action",
        "action_correct",
        "action_result",
        "option_reports",
    )
    return {
        "schema_version": 1,
        "study_id": "V5-RBG-6",
        "trajectory_id": _stable_id(row["condition_id"], *report_order),
        "source_condition_id": row["condition_id"],
        "report_order": list(report_order),
        **{key: row[key] for key in keep},
    }


def dataset_payload(config: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    metadata = source["metadata"]
    if metadata["run_id"] != config["source"]["run_id"]:
        raise ValueError("unexpected source run")
    if metadata["source_dataset_sha256"] != config["source"]["dataset_sha256"]:
        raise ValueError("unexpected source dataset")
    if metadata["model_revision_resolved"] != config["model"]["revision"]:
        raise ValueError("source/model revision mismatch")

    selected = [
        row
        for row in source["rows"]
        if row["incentive"] == "opposed"
        and row["surface_kind"] in {"assertion", "opaque_token"}
    ]
    rows = [
        _source_row(row, order)
        for row in selected
        for order in REPORT_ORDERS
    ]
    body = {
        "schema_version": 1,
        "study_id": "V5-RBG-6",
        "config_sha256": canonical_sha256(config),
        "source_run_id": metadata["run_id"],
        "source_dataset_sha256": metadata["source_dataset_sha256"],
        "rows": rows,
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def verify_dataset_payload(payload: dict[str, Any], config: dict[str, Any]) -> None:
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    if payload.get("content_sha256") != canonical_sha256(body):
        raise ValueError("dataset content hash mismatch")
    if payload.get("config_sha256") != canonical_sha256(config):
        raise ValueError("dataset/config hash mismatch")
    if payload.get("source_run_id") != config["source"]["run_id"]:
        raise ValueError("dataset/source run mismatch")
    if len(payload["rows"]) != int(config["dataset"]["trajectory_contexts"]):
        raise ValueError("unexpected trajectory count")
    if len({row["trajectory_id"] for row in payload["rows"]}) != len(payload["rows"]):
        raise ValueError("trajectory IDs are not unique")
    for row in payload["rows"]:
        if tuple(row["report_order"]) not in REPORT_ORDERS:
            raise ValueError("invalid report order")
        if row["response_per_action"][row["expected_action"]] != row["target_response"]:
            raise ValueError("source oracle action does not reach target")
        if set(expected_reports(row).values()) != set(REPORT_LABELS):
            raise ValueError("report mapping is not bijective")
