from __future__ import annotations

import json

import pytest

from jspace_policy.report_reactivity import (
    COMMON_FINAL_ACTION_QUERY_SUFFIX,
    PRIMARY_ARMS,
    arm_messages,
    dataset_payload,
    independent_fork_queries,
    render_self_report_prompt,
    report_messages,
    self_report_branch,
    verify_dataset_payload,
)


def _config(discovery: int = 4, locked: int = 4) -> dict:
    return {"dataset": {"discovery_bases": discovery, "locked_bases": locked}}


def test_corpus_is_deterministic_and_factorial() -> None:
    config = _config()
    left = dataset_payload(config)
    right = dataset_payload(config)
    assert left == right
    verify_dataset_payload(left, config)
    assert len(left["rows"]) == 8 * 2 * 2 * 2 * 6
    assert {row["split"] for row in left["rows"]} == {"discovery", "locked"}
    assert {row["arm"] for row in left["rows"]} == set(PRIMARY_ARMS)
    assert all(row["prompt"].endswith(COMMON_FINAL_ACTION_QUERY_SUFFIX) for row in left["rows"])
    # These are JSON records, rather than dataclasses or model tensors.
    json.dumps(left, sort_keys=True)


def test_lexical_groups_stay_in_split_and_controls_are_present() -> None:
    rows = dataset_payload(_config(2, 2))["rows"]
    group_splits: dict[str, set[str]] = {}
    for row in rows:
        group_splits.setdefault(row["lexical_group_id"], set()).add(row["split"])
        assert all(
            token not in row["scenario"] for token in ("KITE", "MOSS", "AMBER", "INDIGO")
        )
    assert group_splits and all(len(splits) == 1 for splits in group_splits.values())
    assert {row["surface_kind"] for row in rows} == {"prose", "opaque"}
    assert {row["frame"] for row in rows} == {"strategic", "nonagentic"}


def test_mapping_target_action_and_order_are_balanced() -> None:
    rows = dataset_payload(_config(16, 16))["rows"]
    for split in ("discovery", "locked"):
        bases = [row for row in rows if row["split"] == split and row["arm"] == "direct"]
        assert sum(row["target_index"] == 0 for row in bases) == len(bases) // 2
        assert sum(row["expected_action"] == "A" for row in bases) == len(bases) // 2
        assert sum(row["report_order"] == ["X", "Y"] for row in bases) == len(bases) // 2
        assert sum(row["query_order"] == [0, 1] for row in bases) == len(bases) // 2


def test_oracle_metadata_and_arm_effects_are_content_controlled() -> None:
    rows = dataset_payload(_config(1, 1))["rows"]
    direct = next(row for row in rows if row["arm"] == "direct")
    # Compare through the stable factorial key, rather than relying on row order.
    cell = [
        row
        for row in rows
        if row["base_game_id"] == direct["base_game_id"]
        and row["frame"] == direct["frame"]
        and row["surface_kind"] == direct["surface_kind"]
    ]
    assert {row["arm"] for row in cell} == set(PRIMARY_ARMS)
    oracle = next(row for row in cell if row["arm"] == "oracle")
    swapped = next(row for row in cell if row["arm"] == "swapped")
    assert oracle["oracle"]["expected_action"] == direct["expected_action"]
    assert oracle["arm_report_tokens"] != swapped["arm_report_tokens"]
    assert (
        "external reference card"
        in next(row for row in cell if row["arm"] == "external_facts")["prefix"]
    )
    assert (
        "Formatting check"
        in next(row for row in cell if row["arm"] == "matched_control")["prefix"]
    )


def test_self_report_branch_preserves_actual_tokens_and_marks_accuracy() -> None:
    row = next(
        row for row in dataset_payload(_config(1, 0))["rows"] if row["arm"] == "self_report"
    )
    tokens = [" X ", "Y"]
    branch = self_report_branch(row, tokens)
    assert branch["generated_tokens"] == tokens
    assert branch["report_correct"] == [True, True]
    assert "Answer:\n X " in branch["prompt"]
    assert branch["prompt"].endswith(COMMON_FINAL_ACTION_QUERY_SUFFIX)
    assert render_self_report_prompt(row, tokens) == branch["prompt"]
    with pytest.raises(ValueError):
        self_report_branch(row, ["X"])


def test_independent_forks_share_only_prefix_and_have_capture_points() -> None:
    row = next(row for row in dataset_payload(_config(1, 0))["rows"] if row["arm"] == "direct")
    forks = independent_fork_queries(row)
    assert len(forks["branches"]) == 3
    assert {branch["branch_kind"] for branch in forks["branches"]} == {"action", "report"}
    assert all(branch["recompute_full_prefix"] for branch in forks["branches"])
    assert all(
        branch["shared_prefix_hash"] == forks["shared_prefix_hash"]
        for branch in forks["branches"]
    )
    assert all(
        {point["state_role"] for point in branch["capture_points"]}
        == {"pre_query", "post_query"}
        for branch in forks["branches"]
    )
    assert all(
        branch["prompt"].startswith(forks["shared_prefix"]) for branch in forks["branches"]
    )


def test_chat_messages_keep_reports_as_assistant_turns_and_action_as_user() -> None:
    rows = dataset_payload(_config(1, 0))["rows"]
    self_row = next(row for row in rows if row["arm"] == "self_report")
    messages = arm_messages(self_row, reports=("X", "Y"))
    assert [message["role"] for message in messages] == [
        "user",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert messages[-1]["content"] == COMMON_FINAL_ACTION_QUERY_SUFFIX
    partial = report_messages(self_row, prior_reports=("X",))
    assert partial[-1]["role"] == "user"
    assert partial[-1]["content"] == self_row["report_queries"][1]["question"]
    external_row = next(row for row in rows if row["arm"] == "external_facts")
    external = arm_messages(external_row)
    assert "external reference card" in external[1]["content"]
