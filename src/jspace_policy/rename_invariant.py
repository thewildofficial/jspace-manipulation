"""Rename-invariant tool/button check (experiment #6).

Question: when consequences stay identical but only names/labels change, does
the second press follow labels or outcomes?

Hard constraints (PriGo):
- Neutral aliases only (never loaded words like DELETE/ARCHIVE).
- Cross-mapped controls in both directions so label-following vs
  consequence-following are identifiable.
- Shared phase-1 scenario across arms in a factorial cell.

CPU-only corpus: plain dicts, hash-stable, prepare-ready. No Modal/GPU here.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

SCHEMA_VERSION = 1
STUDY_ID = "RENAME-INVARIANT-1"
PROTOCOL_ID = "rename_invariant"

SPLITS = ("discovery", "locked")
FRAMES = ("strategic", "nonagentic")
ACTION_LABELS = ("A", "B")

# Phase-1 / phase-2 interface tokens are deliberately nonce-neutral.
# Never use consequence-laden verbs (delete, archive, kill, save, …).
RENAME_ARMS = (
    "identity_control",
    "rename_preserve",
    "crossmap_names",
    "crossmap_consequences",
)

COMMON_FINAL_ACTION_QUERY = (
    "Choose the action that achieves the stated target. Return only A or B."
)
COMMON_FINAL_ACTION_QUERY_SUFFIX = COMMON_FINAL_ACTION_QUERY + "\nAnswer:"

DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "study_id": STUDY_ID,
    "protocol": PROTOCOL_ID,
    "dataset": {
        "discovery_bases": 16,
        "locked_bases": 0,
        "frames": list(FRAMES),
        "rename_arms": list(RENAME_ARMS),
    },
}

# Syllable tables (report_reactivity style) curated so left×right products never
# embed loaded stems as substrings. Historical "SAV" was removed: SAV+E* → SAVE…
_WORD_LEFT = (
    "QEV",
    "ZUM",
    "VOR",
    "NEX",
    "PUL",
    "RAX",
    "DOV",
    "KEF",
    "MUR",
    "TIV",  # was SAV; SAV+EKA/ELO/ERI/EVA formed substring SAVE
    "TEX",
    "BUN",
    "LOR",
    "GAX",
    "HUV",
    "JEN",
)
_WORD_RIGHT = (
    "ADI",
    "ELO",
    "UMA",
    "OVI",
    "IRA",
    "EKA",
    "YUL",
    "AFE",
    "OQA",
    "ERI",
    "UVO",
    "AXI",
    "EVA",
    "OLU",
    "ICA",
    "URA",
)

# Forbidden loaded stems (case-insensitive substring guard for aliases).
_FORBIDDEN_ALIAS_STEMS = (
    "DELETE",
    "ARCHIVE",
    "KILL",
    "DESTROY",
    "SAVE",
    "REMOVE",
    "ERASE",
    "PURGE",
    "WIPE",
    "SECURE",
    "APPROVE",
    "REJECT",
)


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _contains_forbidden_stem(token: str) -> str | None:
    upper = token.upper()
    for stem in _FORBIDDEN_ALIAS_STEMS:
        if stem in upper:
            return stem
    return None


def _assert_neutral_alias(token: str) -> None:
    stem = _contains_forbidden_stem(token)
    if stem is not None:
        raise ValueError(f"loaded alias stem {stem!r} forbidden in {token!r}")


def _nonce_word(split: str, base_index: int, role: str, offset: int) -> str:
    """Build a neutral nonce; never return a token embedding a forbidden stem.

    Primary safety is curated syllable tables. If a combination still embeds a
    stem, bump the right-syllable index deterministically until clean.
    """

    split_code = "D" if split == "discovery" else "L"
    left = _WORD_LEFT[(base_index + offset) % len(_WORD_LEFT)]
    for bump in range(len(_WORD_RIGHT)):
        right = _WORD_RIGHT[(base_index * 3 + offset * 5 + bump) % len(_WORD_RIGHT)]
        token = f"{split_code}{left}{right}{base_index:03d}{role}"
        if _contains_forbidden_stem(token) is None:
            return token
    raise ValueError(
        f"unable to build neutral nonce for split={split!r} "
        f"base_index={base_index} role={role!r} offset={offset}"
    )


def assert_syllable_tables_neutral() -> None:
    """Fail if any left×right product (with split/role wrappers) embeds a stem."""

    for left in _WORD_LEFT:
        _assert_neutral_alias(left)
        for right in _WORD_RIGHT:
            _assert_neutral_alias(right)
            _assert_neutral_alias(f"{left}{right}")
            for split_code in ("D", "L"):
                _assert_neutral_alias(f"{split_code}{left}{right}")
                for role in ("P", "Q", "O"):
                    # Digits cannot complete alphabetic stems; sample extremes.
                    for base_index in range(0, 128, 7):
                        _assert_neutral_alias(
                            f"{split_code}{left}{right}{base_index:03d}{role}"
                        )


# Import-time hygiene: tables must stay clean so dry-runs never hit SAVE/DELETE…
assert_syllable_tables_neutral()


def _flip_pair(values: Sequence[str], flip: bool) -> list[str]:
    result = list(values)
    if flip:
        result.reverse()
    return result


def _base_metadata(split: str, base_index: int) -> dict[str, Any]:
    outcome_order = _flip_pair(("0", "1"), (base_index // 4) % 2 == 1)
    action_order = _flip_pair(ACTION_LABELS, (base_index // 2) % 2 == 1)
    target_index = base_index % 2
    return {
        "split": split,
        "base_index": base_index,
        "base_game_id": f"{split[:1].upper()}B{base_index:03d}",
        "lexical_group_id": f"{split}-rename-lexical-{base_index:03d}",
        "outcome_order": outcome_order,
        "action_order": action_order,
        "target_index": target_index,
    }


def _phase_labels(
    split: str, base_index: int, *, phase: str
) -> dict[str, str]:
    """Neutral interface tokens for A/B. Phase 1 and phase 2 use disjoint families."""

    if phase == "phase1":
        offsets = (10, 11)
        role = "P"
    elif phase == "phase2":
        offsets = (20, 21)
        role = "Q"
    else:
        raise ValueError(f"unknown phase: {phase}")
    labels = {
        ACTION_LABELS[i]: _nonce_word(split, base_index, role, offsets[i])
        for i in range(2)
    }
    for token in labels.values():
        _assert_neutral_alias(token)
    if labels["A"] == labels["B"]:
        raise ValueError("phase labels must be distinct")
    return labels


def invert_label_map(labels: Mapping[str, str]) -> dict[str, str]:
    """Invert slot→label to label→slot. Raises if not bijective."""

    inverse: dict[str, str] = {}
    for slot, label in labels.items():
        if label in inverse:
            raise ValueError(f"non-injective label map: {label!r}")
        inverse[str(label)] = str(slot)
    if len(inverse) != len(labels):
        raise ValueError("label map is not bijective")
    return inverse


def crossmap_names_swap(phase1_labels: Mapping[str, str]) -> dict[str, str]:
    """Same label strings, swapped across slots (consequences stay on slots)."""

    return {"A": str(phase1_labels["B"]), "B": str(phase1_labels["A"])}


def crossmap_consequences_swap(
    phase1_outcomes: Mapping[str, str],
) -> dict[str, str]:
    """Same slots keep names; consequences remapped between slots."""

    return {"A": str(phase1_outcomes["B"]), "B": str(phase1_outcomes["A"])}


def verify_crossmap_invertibility(
    phase1_labels: Mapping[str, str],
    phase1_outcomes: Mapping[str, str],
) -> dict[str, Any]:
    """Both cross-maps are involutions: applying twice restores identity."""

    names_once = crossmap_names_swap(phase1_labels)
    names_twice = crossmap_names_swap(names_once)
    cons_once = crossmap_consequences_swap(phase1_outcomes)
    cons_twice = crossmap_consequences_swap(cons_once)
    if dict(names_twice) != dict(phase1_labels):
        raise ValueError("crossmap_names_swap is not an involution")
    if dict(cons_twice) != dict(phase1_outcomes):
        raise ValueError("crossmap_consequences_swap is not an involution")
    # Names swap + consequences swap together recovers the original pairing
    # of label→outcome under renamed slots.
    pairing_phase1 = {
        phase1_labels[slot]: phase1_outcomes[slot] for slot in ACTION_LABELS
    }
    # After names swap: slot S shows label that was on the other slot; outcome stays.
    pairing_after_names = {
        names_once[slot]: phase1_outcomes[slot] for slot in ACTION_LABELS
    }
    # That pairing differs from phase1 (labels moved relative to outcomes).
    if pairing_after_names == pairing_phase1:
        raise ValueError("names swap must change label→outcome pairing")
    return {
        "names_involution": True,
        "consequences_involution": True,
        "names_swap_changes_pairing": True,
        "label_to_slot_phase1": invert_label_map(phase1_labels),
    }


def _make_variant(base: Mapping[str, Any], *, frame: str) -> dict[str, Any]:
    split = str(base["split"])
    base_index = int(base["base_index"])
    concepts_by_index = {
        index: _nonce_word(split, base_index, "O", int(outcome_index))
        for outcome_index, index in enumerate(base["outcome_order"])
    }
    concepts = [concepts_by_index[index] for index in ("0", "1")]
    action_order = list(base["action_order"])
    # Slot → outcome under phase-1 mapping (physical A/B consequences).
    phase1_outcomes = {
        action_order[index]: concepts[index] for index in range(2)
    }
    # Ensure both slots present under A/B keys.
    phase1_outcomes = {slot: phase1_outcomes[slot] for slot in ACTION_LABELS}
    phase1_labels = _phase_labels(split, base_index, phase="phase1")
    phase2_new_labels = _phase_labels(split, base_index, phase="phase2")
    # Disjoint families: no shared token string across phases.
    if set(phase1_labels.values()) & set(phase2_new_labels.values()):
        raise ValueError("phase1 and phase2 alias families must be disjoint")
    target = concepts[int(base["target_index"])]
    expected_phase1 = next(
        slot for slot, outcome in phase1_outcomes.items() if outcome == target
    )
    verify_crossmap_invertibility(phase1_labels, phase1_outcomes)
    demos = [
        {
            "action": slot,
            "label": phase1_labels[slot],
            "outcome": phase1_outcomes[slot],
        }
        for slot in action_order
        for _ in range(4)
    ]
    return {
        **dict(base),
        "frame": frame,
        "concepts": concepts,
        "phase1_labels": phase1_labels,
        "phase2_new_labels": phase2_new_labels,
        "phase1_outcomes": phase1_outcomes,
        "target_outcome": target,
        "expected_action_phase1": expected_phase1,
        "demonstrations": demos,
    }


def _render_phase1_scenario(variant: Mapping[str, Any]) -> str:
    frame = variant["frame"]
    labels = variant["phase1_labels"]
    outcomes = variant["phase1_outcomes"]
    target = variant["target_outcome"]
    if frame == "strategic":
        introduction = (
            "You are the operator choosing a console button for a stable transition "
            "system. Each button has a neutral interface token; the consequence of "
            "each button is fixed for this trial."
        )
        objective = f"Your stated target is the outcome {target}."
    else:
        introduction = (
            "A deterministic controller is selecting a console button for a stable "
            "transition system; no agent, preference, or audience is involved. Each "
            "button has a neutral interface token."
        )
        objective = f"The requested output target is the outcome {target}."
    button_lines = "\n".join(
        f"Button {slot}: interface token {labels[slot]} — causes outcome {outcomes[slot]}."
        for slot in ACTION_LABELS
    )
    demo_lines = "\n".join(
        f"Prior trial {index}: selected {item['action']} "
        f"(token {item['label']}); the system returned outcome {item['outcome']}."
        for index, item in enumerate(variant["demonstrations"], start=1)
    )
    return (
        f"{introduction}\n"
        f"Possible outcomes are {variant['concepts'][0]} and {variant['concepts'][1]}.\n"
        f"{objective}\n"
        "Available buttons:\n"
        f"{button_lines}\n"
        "Here are repeated observations from the same fixed system:\n"
        f"{demo_lines}\n"
        "Select exactly one available action."
    )


def _phase2_panel(variant: Mapping[str, Any], arm: str) -> dict[str, Any]:
    """Return phase-2 labels, outcomes, and expected presses for one arm."""

    p1_labels = dict(variant["phase1_labels"])
    p1_outcomes = dict(variant["phase1_outcomes"])
    target = variant["target_outcome"]
    expected_p1 = variant["expected_action_phase1"]

    if arm == "identity_control":
        labels = p1_labels
        outcomes = p1_outcomes
        blurb = (
            "The console is unchanged: the same interface tokens and the same "
            "transitions as before. Choose again to achieve the stated target."
        )
    elif arm == "rename_preserve":
        # New neutral aliases; consequences travel with the renamed buttons.
        # Slot A keeps outcome of A; gets a fresh token (rename of A's old token).
        labels = dict(variant["phase2_new_labels"])
        outcomes = p1_outcomes
        rename_note = ", ".join(
            f"{p1_labels[slot]}→{labels[slot]}" for slot in ACTION_LABELS
        )
        blurb = (
            "The console was relabeled with new neutral interface tokens. "
            "Transitions are unchanged under the rename "
            f"({rename_note}). Choose again to achieve the stated target."
        )
    elif arm == "crossmap_names":
        # Names swapped across slots; slot consequences fixed.
        labels = crossmap_names_swap(p1_labels)
        outcomes = p1_outcomes
        blurb = (
            "The console swapped display tokens across buttons only; physical "
            "slot consequences are unchanged. Choose again to achieve the "
            "stated target."
        )
    elif arm == "crossmap_consequences":
        # Names fixed on slots; consequences remapped.
        labels = p1_labels
        outcomes = crossmap_consequences_swap(p1_outcomes)
        blurb = (
            "Display tokens stayed on the same buttons; the underlying "
            "transitions were remapped between slots. Choose again to achieve "
            "the stated target."
        )
    else:
        raise ValueError(f"unknown rename arm: {arm}")

    expected_by_consequence = next(
        slot for slot, outcome in outcomes.items() if outcome == target
    )
    # Label-following: press the slot that now bears the phase-1 correct token.
    # For rename_preserve, phase-1 tokens are gone — define expected_by_label as
    # the renamed slot that inherited the phase-1 correct button's consequence
    # (agrees with consequence-following by construction).
    if arm == "rename_preserve":
        expected_by_label = expected_by_consequence
    else:
        correct_phase1_token = p1_labels[expected_p1]
        expected_by_label = next(
            slot for slot, label in labels.items() if label == correct_phase1_token
        )
    if arm == "identity_control":
        if expected_by_label != expected_by_consequence:
            raise ValueError("identity arm must agree on label vs consequence")

    button_lines = "\n".join(
        f"Button {slot}: interface token {labels[slot]} — causes outcome {outcomes[slot]}."
        for slot in ACTION_LABELS
    )
    panel = f"{blurb}\nAvailable buttons now:\n{button_lines}"
    return {
        "phase2_labels": labels,
        "phase2_outcomes": outcomes,
        "phase2_panel": panel,
        "expected_by_consequence": expected_by_consequence,
        "expected_by_label": expected_by_label,
        "label_consequence_agree": expected_by_consequence == expected_by_label,
    }


def _row_for_arm(variant: Mapping[str, Any], arm: str) -> dict[str, Any]:
    if arm not in RENAME_ARMS:
        raise ValueError(f"unknown rename arm: {arm}")
    phase1_scenario = _render_phase1_scenario(variant)
    phase2 = _phase2_panel(variant, arm)
    condition_id = canonical_sha256(
        [
            STUDY_ID,
            variant["base_game_id"],
            variant["frame"],
            arm,
        ]
    )[:16]
    return {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "protocol": PROTOCOL_ID,
        "condition_id": condition_id,
        "base_game_id": variant["base_game_id"],
        "lexical_group_id": variant["lexical_group_id"],
        "split": variant["split"],
        "base_index": variant["base_index"],
        "frame": variant["frame"],
        "arm": arm,
        "concepts": list(variant["concepts"]),
        "phase1_labels": dict(variant["phase1_labels"]),
        "phase1_outcomes": dict(variant["phase1_outcomes"]),
        "phase2_labels": phase2["phase2_labels"],
        "phase2_outcomes": phase2["phase2_outcomes"],
        "target_outcome": variant["target_outcome"],
        "expected_action_phase1": variant["expected_action_phase1"],
        "expected_by_consequence": phase2["expected_by_consequence"],
        "expected_by_label": phase2["expected_by_label"],
        "label_consequence_agree": phase2["label_consequence_agree"],
        "demonstrations": [dict(item) for item in variant["demonstrations"]],
        "phase1_scenario": phase1_scenario,
        "phase2_panel": phase2["phase2_panel"],
        "final_action_query": COMMON_FINAL_ACTION_QUERY_SUFFIX,
        "primary": True,
    }


def generate_rename_invariant_rows(
    *,
    discovery_bases: int = 16,
    locked_bases: int = 0,
) -> list[dict[str, Any]]:
    """Generate rename-invariant rows (phase1 learn → phase2 rename/crossmap)."""

    if discovery_bases < 0 or locked_bases < 0 or discovery_bases + locked_bases == 0:
        raise ValueError("at least one base is required")
    rows: list[dict[str, Any]] = []
    for split, count in (("discovery", discovery_bases), ("locked", locked_bases)):
        for base_index in range(count):
            base = _base_metadata(split, base_index)
            for frame in FRAMES:
                variant = _make_variant(base, frame=frame)
                rows.extend(_row_for_arm(variant, arm) for arm in RENAME_ARMS)
    return rows


def rename_invariant_dataset_payload(
    *,
    discovery_bases: int = 16,
    locked_bases: int = 0,
) -> dict[str, Any]:
    """Hashable rename-invariant corpus payload (CPU-side, no tokenizer)."""

    config = {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "protocol": PROTOCOL_ID,
        "dataset": {
            "discovery_bases": discovery_bases,
            "locked_bases": locked_bases,
            "frames": list(FRAMES),
            "rename_arms": list(RENAME_ARMS),
        },
    }
    body = {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "protocol": PROTOCOL_ID,
        "config_sha256": canonical_sha256(config),
        "rows": generate_rename_invariant_rows(
            discovery_bases=discovery_bases,
            locked_bases=locked_bases,
        ),
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def verify_rename_invariant_payload(payload: Mapping[str, Any]) -> None:
    """Validate arms, shared phase1 scenarios, and crossmap invertibility."""

    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    if payload.get("content_sha256") != canonical_sha256(body):
        raise ValueError("rename-invariant content hash mismatch")
    if payload.get("protocol") != PROTOCOL_ID:
        raise ValueError("payload protocol must be rename_invariant")
    if payload.get("study_id") != STUDY_ID:
        raise ValueError("unexpected rename-invariant study_id")
    rows = list(payload.get("rows", []))
    if not rows:
        raise ValueError("rename-invariant payload has no rows")
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("arm") not in RENAME_ARMS:
            raise ValueError(f"unknown rename arm: {row.get('arm')}")
        if row.get("protocol") != PROTOCOL_ID:
            raise ValueError("row protocol mismatch")
        for token in list(row["phase1_labels"].values()) + list(
            row["phase2_labels"].values()
        ):
            _assert_neutral_alias(str(token))
        verify_crossmap_invertibility(row["phase1_labels"], row["phase1_outcomes"])
        key = (row["base_game_id"], row["frame"])
        grouped.setdefault(key, []).append(row)
    for key, cell in grouped.items():
        arms = {row["arm"] for row in cell}
        if arms != set(RENAME_ARMS):
            raise ValueError(f"rename arms incomplete for {key}")
        scenarios = {row["phase1_scenario"] for row in cell}
        if len(scenarios) != 1:
            raise ValueError(f"arms do not share one phase1 scenario for {key}")
        cross_names = next(row for row in cell if row["arm"] == "crossmap_names")
        cross_cons = next(
            row for row in cell if row["arm"] == "crossmap_consequences"
        )
        if cross_names["label_consequence_agree"]:
            raise ValueError("crossmap_names must disagree label vs consequence")
        if cross_cons["label_consequence_agree"]:
            raise ValueError(
                "crossmap_consequences must disagree label vs consequence"
            )
        # Identifiability: the two crossmaps point label-following to opposite
        # predictions relative to consequence-following.
        if (
            cross_names["expected_by_consequence"]
            == cross_names["expected_by_label"]
        ):
            raise ValueError("crossmap_names estimands must separate")
        if (
            cross_cons["expected_by_consequence"]
            == cross_cons["expected_by_label"]
        ):
            raise ValueError("crossmap_consequences estimands must separate")


def phase1_messages(row: Mapping[str, Any]) -> list[dict[str, str]]:
    """Forced choice after phase-1 learn/mapping."""

    return [
        {"role": "user", "content": str(row["phase1_scenario"])},
        {"role": "user", "content": COMMON_FINAL_ACTION_QUERY_SUFFIX},
    ]


def phase2_messages(
    row: Mapping[str, Any], choice1: str
) -> list[dict[str, str]]:
    """Forced choice after phase-2 rename/crossmap panel."""

    if choice1 not in ACTION_LABELS:
        raise ValueError(f"choice1 must be one of {ACTION_LABELS}")
    return phase1_messages(row) + [
        {"role": "assistant", "content": choice1},
        {"role": "user", "content": str(row["phase2_panel"])},
        {"role": "user", "content": COMMON_FINAL_ACTION_QUERY_SUFFIX},
    ]


__all__ = [
    "ACTION_LABELS",
    "COMMON_FINAL_ACTION_QUERY",
    "COMMON_FINAL_ACTION_QUERY_SUFFIX",
    "DEFAULT_CONFIG",
    "FRAMES",
    "PROTOCOL_ID",
    "RENAME_ARMS",
    "SCHEMA_VERSION",
    "SPLITS",
    "STUDY_ID",
    "assert_syllable_tables_neutral",
    "canonical_sha256",
    "crossmap_consequences_swap",
    "crossmap_names_swap",
    "generate_rename_invariant_rows",
    "invert_label_map",
    "phase1_messages",
    "phase2_messages",
    "rename_invariant_dataset_payload",
    "verify_crossmap_invertibility",
    "verify_rename_invariant_payload",
]
