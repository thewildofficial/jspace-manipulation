"""Rename-invariant corpus: hash stability and cross-map invertibility."""

from __future__ import annotations

import pytest

from jspace_policy import rename_invariant as rename_mod
from jspace_policy.rename_invariant import (
    ACTION_LABELS,
    COMMON_FINAL_ACTION_QUERY_SUFFIX,
    RENAME_ARMS,
    assert_syllable_tables_neutral,
    crossmap_consequences_swap,
    crossmap_names_swap,
    generate_rename_invariant_rows,
    invert_label_map,
    phase1_messages,
    phase2_messages,
    rename_invariant_dataset_payload,
    verify_crossmap_invertibility,
    verify_rename_invariant_payload,
)


def test_rename_invariant_hash_stable_and_arms_share_phase1() -> None:
    left = rename_invariant_dataset_payload(discovery_bases=2, locked_bases=0)
    right = rename_invariant_dataset_payload(discovery_bases=2, locked_bases=0)
    assert left == right
    assert left["content_sha256"] == right["content_sha256"]
    verify_rename_invariant_payload(left)
    assert {row["arm"] for row in left["rows"]} == set(RENAME_ARMS)
    assert all(row["protocol"] == "rename_invariant" for row in left["rows"])
    # 2 bases × 2 frames × 4 arms
    assert len(left["rows"]) == 2 * 2 * 4

    sample = left["rows"][0]
    cell = [
        row
        for row in left["rows"]
        if row["base_game_id"] == sample["base_game_id"]
        and row["frame"] == sample["frame"]
    ]
    assert {row["arm"] for row in cell} == set(RENAME_ARMS)
    assert len({row["phase1_scenario"] for row in cell}) == 1


def test_crossmap_invertibility_and_separation() -> None:
    row = next(
        r
        for r in generate_rename_invariant_rows(discovery_bases=1, locked_bases=0)
        if r["arm"] == "identity_control"
    )
    check = verify_crossmap_invertibility(
        row["phase1_labels"], row["phase1_outcomes"]
    )
    assert check["names_involution"]
    assert check["consequences_involution"]
    names = crossmap_names_swap(row["phase1_labels"])
    assert names == {"A": row["phase1_labels"]["B"], "B": row["phase1_labels"]["A"]}
    assert crossmap_names_swap(names) == dict(row["phase1_labels"])
    cons = crossmap_consequences_swap(row["phase1_outcomes"])
    assert crossmap_consequences_swap(cons) == dict(row["phase1_outcomes"])
    inverse = invert_label_map(row["phase1_labels"])
    assert {inverse[label] for label in row["phase1_labels"].values()} == set(
        ACTION_LABELS
    )

    rows = generate_rename_invariant_rows(discovery_bases=1, locked_bases=0)
    for arm in ("crossmap_names", "crossmap_consequences"):
        cross = next(r for r in rows if r["arm"] == arm)
        assert not cross["label_consequence_agree"]
        assert cross["expected_by_consequence"] != cross["expected_by_label"]
    preserve = next(r for r in rows if r["arm"] == "rename_preserve")
    assert preserve["label_consequence_agree"]
    identity = next(r for r in rows if r["arm"] == "identity_control")
    assert identity["label_consequence_agree"]


def test_syllable_tables_cannot_form_forbidden_stems() -> None:
    """Regression: SAV+EKA produced DSAVEKA004Q and broke Actions prepare."""

    assert_syllable_tables_neutral()
    assert "SAV" not in rename_mod._WORD_LEFT
    # Exhaustive left×right products (the dry-run failure mode).
    for left in rename_mod._WORD_LEFT:
        for right in rename_mod._WORD_RIGHT:
            concat = f"{left}{right}".upper()
            for stem in rename_mod._FORBIDDEN_ALIAS_STEMS:
                assert stem not in concat, (left, right, stem)


def test_neutral_aliases_forbid_loaded_stems_many_bases() -> None:
    """Sample well beyond the 16-base pilot so substring collisions cannot hide."""

    rows = generate_rename_invariant_rows(discovery_bases=48, locked_bases=16)
    assert len(rows) == (48 + 16) * 2 * 4
    forbidden = rename_mod._FORBIDDEN_ALIAS_STEMS
    seen_tokens: set[str] = set()
    for row in rows:
        tokens = [
            *row["phase1_labels"].values(),
            *row["phase2_labels"].values(),
            *row["concepts"],
        ]
        for token in tokens:
            seen_tokens.add(token)
            upper = token.upper()
            for stem in forbidden:
                assert stem not in upper, (token, stem, row["base_game_id"])
    # Specific regression token from Actions 34053946547 must not reappear.
    assert "DSAVEKA004Q" not in seen_tokens
    assert not any("SAVE" in t.upper() for t in seen_tokens)


def test_phase1_and_phase2_messages_share_final_instruction() -> None:
    row = next(
        r
        for r in generate_rename_invariant_rows(discovery_bases=1, locked_bases=0)
        if r["arm"] == "rename_preserve"
    )
    p1 = phase1_messages(row)
    p2 = phase2_messages(row, "A")
    assert p1[-1]["content"] == COMMON_FINAL_ACTION_QUERY_SUFFIX
    assert p2[-1]["content"] == COMMON_FINAL_ACTION_QUERY_SUFFIX
    assert p1 == p2[:2]
    assert p2[2]["content"] == "A"
    assert "interface token" in p2[3]["content"]
    # Phase2 uses fresh tokens for rename_preserve.
    for token in row["phase2_labels"].values():
        assert token in p2[3]["content"]
    for token in row["phase1_labels"].values():
        assert token not in p2[3]["content"] or "→" in p2[3]["content"]


def test_rename_preserve_phase_families_disjoint() -> None:
    row = next(
        r
        for r in generate_rename_invariant_rows(discovery_bases=1, locked_bases=0)
        if r["arm"] == "rename_preserve"
    )
    assert set(row["phase1_labels"].values()).isdisjoint(
        set(row["phase2_labels"].values())
    )


def test_invert_label_map_rejects_collision() -> None:
    with pytest.raises(ValueError, match="non-injective"):
        invert_label_map({"A": "SAME", "B": "SAME"})


def test_assert_neutral_alias_still_rejects_save_substring() -> None:
    with pytest.raises(ValueError, match="SAVE"):
        rename_mod._assert_neutral_alias("DSAVEKA004Q")
