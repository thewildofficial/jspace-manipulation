"""Fresh, deterministic prompt corpus for the report-reactivity experiment.

The corpus is deliberately kept as plain dictionaries.  This makes it safe to
prepare on CPU, hash before a model run, and move through JSONL without relying
on a model-specific object or a live tokenizer.  A row is one arm of one game
variant; all six primary arms share the same scenario and final action query.

The generated lexical items are nonce words.  They are not copied from the
older RBG corpora, and a lexical group (a base game) never crosses the
discovery/locked split.  ``self_report_branch`` is used after generation has
produced the two actual report tokens; it intentionally preserves those tokens
verbatim in the action prompt.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

SCHEMA_VERSION = 1
STUDY_ID = "REPORT-REACTIVITY-1"
# Canonical protocol ID doubles as study_id (see experiments/report_reactivity/NAMING.md).
# Historical payloads used ASK-MID-TRAJECTORY-1; do not rename committed results dirs.
MID_TRAJECTORY_STUDY_ID = "ask_mid_trajectory"
MID_TRAJECTORY_LEGACY_STUDY_ID = "ASK-MID-TRAJECTORY-1"

SPLITS = ("discovery", "locked")
FRAMES = ("strategic", "nonagentic")
SURFACE_KINDS = ("prose", "opaque")
POLICY_KINDS = ("opposed_assertion", "direct_policy")
PRIMARY_ARMS = (
    "direct",
    "self_report",
    "oracle",
    "swapped",
    "matched_control",
    "external_facts",
)
# Mid-trajectory ask-as-intervention arms: forced choice → mid turn → forced choice.
# Default six-arm report corpus is unchanged; these arms are generated only by
# ``generate_mid_trajectory_rows``.
MID_TRAJECTORY_ARMS = (
    "mid_no_ask_control",
    "mid_ask_self",
    "mid_ask_oracle",
    "mid_ask_swapped",
)
ACTION_LABELS = ("A", "B")
REPORT_LABELS = ("X", "Y")

# History modes control how many correct prior-trial demonstrations appear.
# ``minimal`` is the historical default (four repeats of each mapped action).
# ``redundant`` adds two extra full correct cycles (inverse-evidence style) to
# break Direct-ceiling pilots without changing lexicon, arms, or the final query.
HISTORY_MODES = ("minimal", "redundant")
DEFAULT_HISTORY_MODE = "minimal"
MINIMAL_DEMO_REPEATS = 4
REDUNDANT_EXTRA_CYCLES = 2

# The exact same string is appended to every primary prompt.  Keep this as a
# public constant so a runner can assert suffix equality before dispatch.
COMMON_FINAL_ACTION_QUERY = (
    "Choose the action that achieves the stated target. Return only A or B."
)
COMMON_FINAL_ACTION_QUERY_SUFFIX = COMMON_FINAL_ACTION_QUERY + "\nAnswer:"

DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "study_id": STUDY_ID,
    "history_mode": DEFAULT_HISTORY_MODE,
    "dataset": {
        "discovery_bases": 48,
        "locked_bases": 96,
        "frames": list(FRAMES),
        "surface_kinds": list(SURFACE_KINDS),
        "policy_kinds": list(POLICY_KINDS),
        "primary_arms": list(PRIMARY_ARMS),
    },
}

# Every item contains an index suffix, so a future edit to the syllable table
# cannot silently create a lexical collision.  These strings are intentionally
# unlike the concept pairs used by the historical game modules.
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
    "SAV",
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


def canonical_sha256(value: object) -> str:
    """Return the stable hash used by corpus manifests."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _nonce_word(split: str, base_index: int, role: str, offset: int) -> str:
    split_code = "D" if split == "discovery" else "L"
    left = _WORD_LEFT[(base_index + offset) % len(_WORD_LEFT)]
    right = _WORD_RIGHT[(base_index * 3 + offset * 5) % len(_WORD_RIGHT)]
    return f"{split_code}{left}{right}{base_index:03d}{role}"


def _flip_pair(values: Sequence[str], flip: bool) -> list[str]:
    result = list(values)
    if flip:
        result.reverse()
    return result


def _normalise_history_mode(value: object) -> str:
    mode = str(value if value is not None else DEFAULT_HISTORY_MODE)
    if mode not in HISTORY_MODES:
        raise ValueError(f"history_mode must be one of {HISTORY_MODES}")
    return mode


def _demo_repeats(history_mode: str) -> int:
    mode = _normalise_history_mode(history_mode)
    if mode == "minimal":
        return MINIMAL_DEMO_REPEATS
    return MINIMAL_DEMO_REPEATS + REDUNDANT_EXTRA_CYCLES


def _normalise_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    supplied = config or DEFAULT_CONFIG
    dataset = supplied.get("dataset", supplied)
    discovery = int(dataset.get("discovery_bases", dataset.get("discovery", 48)))
    locked = int(dataset.get("locked_bases", dataset.get("locked", 96)))
    if discovery < 0 or locked < 0 or discovery + locked == 0:
        raise ValueError("at least one non-negative discovery or locked base is required")
    frames = tuple(dataset.get("frames", FRAMES))
    surfaces = tuple(dataset.get("surface_kinds", SURFACE_KINDS))
    policies = tuple(dataset.get("policy_kinds", POLICY_KINDS))
    arms = tuple(dataset.get("primary_arms", PRIMARY_ARMS))
    history_mode = _normalise_history_mode(
        supplied.get("history_mode", dataset.get("history_mode", DEFAULT_HISTORY_MODE))
    )
    if set(frames) != set(FRAMES):
        raise ValueError(f"frames must be exactly {FRAMES}")
    if set(surfaces) != set(SURFACE_KINDS):
        raise ValueError(f"surface_kinds must be exactly {SURFACE_KINDS}")
    if set(policies) != set(POLICY_KINDS):
        raise ValueError(f"policy_kinds must be exactly {POLICY_KINDS}")
    if set(arms) != set(PRIMARY_ARMS):
        raise ValueError(f"primary_arms must be exactly {PRIMARY_ARMS}")
    return {
        "schema_version": int(supplied.get("schema_version", SCHEMA_VERSION)),
        "study_id": supplied.get("study_id", STUDY_ID),
        "history_mode": history_mode,
        "dataset": {
            "discovery_bases": discovery,
            "locked_bases": locked,
            "frames": list(FRAMES),
            "surface_kinds": list(SURFACE_KINDS),
            "policy_kinds": list(POLICY_KINDS),
            "primary_arms": list(PRIMARY_ARMS),
        },
    }


def _base_metadata(split: str, base_index: int) -> dict[str, Any]:
    # The four independent parity schedules balance the factors at the proposed
    # 48/96 sizes, while retaining enough within-base counterbalancing to use a
    # smaller pilot (for example 16 bases per split).
    outcome_order = _flip_pair(("0", "1"), (base_index // 4) % 2 == 1)
    action_order = _flip_pair(ACTION_LABELS, (base_index // 2) % 2 == 1)
    report_order = _flip_pair(REPORT_LABELS, (base_index // 8) % 2 == 1)
    query_order = [0, 1]
    if base_index % 2 == 1:
        query_order.reverse()
    target_index = base_index % 2
    return {
        "split": split,
        "base_index": base_index,
        "base_game_id": f"{split[:1].upper()}B{base_index:03d}",
        "lexical_group_id": f"{split}-lexical-{base_index:03d}",
        "outcome_order": outcome_order,
        "action_order": action_order,
        "report_order": report_order,
        "query_order": query_order,
        "target_index": target_index,
    }


def _render_scenario(
    *,
    frame: str,
    surface_kind: str,
    concepts: list[str],
    action_to_outcome: dict[str, str],
    action_text: dict[str, str],
    target: str,
    demonstrations: Sequence[Mapping[str, str]],
) -> str:
    if frame == "strategic":
        introduction = (
            "You are the operator choosing an action for a stable transition system. "
            "The consequence of each action is fixed for this trial."
        )
        objective = f"Your stated target is the outcome {target}."
    else:
        introduction = (
            "A deterministic controller is selecting an input for a stable transition "
            "system; no agent, preference, or audience is involved."
        )
        objective = f"The requested output target is the outcome {target}."
    demonstration_lines = "\n".join(
        f"Prior trial {index}: selected {item['action']} ({action_text[item['action']]}); "
        f"the system returned outcome {item['outcome']}."
        for index, item in enumerate(demonstrations, start=1)
    )
    return (
        f"{introduction}\n"
        f"Possible outcomes are {concepts[0]} and {concepts[1]}.\n"
        f"{objective}\n"
        "Here are repeated observations from the same fixed system:\n"
        f"{demonstration_lines}\n"
        "Select exactly one available action."
    )


def _report_question(
    *, row: Mapping[str, Any], action: str, question_index: int
) -> tuple[str, str]:
    action_text = row["action_text"][action]
    concepts = row["concepts"]
    labels = row["report_mapping"]
    question = (
        f"Consequence check {question_index + 1}: for action {action} ({action_text}), "
        f"which outcome follows? Options: {labels[concepts[0]]}={concepts[0]}; "
        f"{labels[concepts[1]]}={concepts[1]}. Return only X or Y."
    )
    return question, labels[row["action_to_outcome"][action]]


def _report_queries(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    for query_index, action_index in enumerate(row["query_order"]):
        action = row["action_order"][action_index]
        question, expected = _report_question(
            row=row, action=action, question_index=query_index
        )
        queries.append(
            {
                "query_index": query_index,
                "action": action,
                "question": question,
                "expected_token": expected,
                "expected_outcome": row["action_to_outcome"][action],
            }
        )
    return queries


def _control_queries(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    # These questions have the same forced-choice shape and comparable length,
    # but never ask about an action's consequence.
    answers = list(row["report_order"])
    queries: list[dict[str, Any]] = []
    for query_index, answer_index in enumerate(row["query_order"]):
        expected = answers[answer_index]
        question = (
            f"Formatting check {query_index + 1}: which marker is requested in this "
            f"control turn, {expected} or its alternate? Return only X or Y."
        )
        queries.append(
            {
                "query_index": query_index,
                "question": question,
                "expected_token": expected,
            }
        )
    return queries


def _turns_with_answers(
    queries: Sequence[Mapping[str, Any]], answers: Sequence[str] | None, *, title: str
) -> str:
    pieces = [title]
    for index, query in enumerate(queries):
        pieces.append(str(query["question"]))
        pieces.append("Answer:")
        if answers is not None:
            pieces.append(str(answers[index]))
    return "\n".join(pieces)


def _make_variant(
    base: Mapping[str, Any],
    *,
    frame: str,
    surface_kind: str,
    policy_kind: str,
    history_mode: str = DEFAULT_HISTORY_MODE,
) -> dict[str, Any]:
    split = str(base["split"])
    base_index = int(base["base_index"])
    history_mode = _normalise_history_mode(history_mode)
    concepts_by_index = {
        index: _nonce_word(split, base_index, "O", int(outcome_index))
        for outcome_index, index in enumerate(base["outcome_order"])
    }
    concepts = [concepts_by_index[index] for index in ("0", "1")]
    action_order = list(base["action_order"])
    action_to_outcome = {action_order[index]: concepts[index] for index in range(2)}
    opaque_tokens = {
        action_order[index]: _nonce_word(split, base_index, "T", index + 2)
        for index in range(2)
    }
    if surface_kind == "opaque":
        action_text = {
            action: f"emit opaque signal {opaque_tokens[action]}" for action in ACTION_LABELS
        }
    elif policy_kind == "opposed_assertion":
        action_text = {
            action: f"assert that outcome {concepts[1 - concepts.index(outcome)]}"
            for action, outcome in action_to_outcome.items()
        }
    else:
        action_text = {
            action: f"cause outcome {outcome}" for action, outcome in action_to_outcome.items()
        }
    target = concepts[int(base["target_index"])]
    report_mapping = {concepts[index]: base["report_order"][index] for index in range(2)}
    variant = {
        **dict(base),
        "frame": frame,
        "surface_kind": surface_kind,
        "policy_kind": policy_kind,
        "history_mode": history_mode,
        "concepts": concepts,
        "opaque_tokens": opaque_tokens,
        "action_text": action_text,
        "action_to_outcome": action_to_outcome,
        "target_outcome": target,
        "report_mapping": report_mapping,
        "expected_action": action_order[int(base["target_index"])],
    }
    demo_action_order = [action_order[index] for index in base["query_order"]]
    # minimal: four repeats per mapped action (historical). redundant: two extra
    # full correct cycles (each action appears two more times), still correct.
    demonstrations = [
        {"action": action, "outcome": action_to_outcome[action]}
        for action in demo_action_order
        for _ in range(_demo_repeats(history_mode))
    ]
    variant["demonstrations"] = demonstrations
    variant["report_queries"] = _report_queries(variant)
    variant["control_queries"] = _control_queries(variant)
    variant["expected_report_tokens"] = [
        query["expected_token"] for query in variant["report_queries"]
    ]
    variant["swapped_report_tokens"] = [
        REPORT_LABELS[1 - REPORT_LABELS.index(token)]
        for token in variant["expected_report_tokens"]
    ]
    variant["oracle"] = {
        "policy_kind": policy_kind,
        "target_outcome": target,
        "expected_action": variant["expected_action"],
        "action_to_outcome": dict(action_to_outcome),
        "outcome_to_report": dict(report_mapping),
        "report_query_order": list(base["query_order"]),
        "expected_report_tokens": list(variant["expected_report_tokens"]),
    }
    return variant


def _arm_prefix(row: Mapping[str, Any], arm: str, answers: Sequence[str] | None = None) -> str:
    scenario = str(row["scenario"])
    if arm == "direct":
        return scenario
    if arm == "self_report":
        turns = _turns_with_answers(
            row["report_queries"],
            answers,
            title="Before acting, answer two private consequence checks.",
        )
        return f"{scenario}\n\n{turns}"
    if arm == "oracle":
        turns = _turns_with_answers(
            row["report_queries"],
            row["expected_report_tokens"],
            title="A verified report gives two consequence checks.",
        )
        return f"{scenario}\n\n{turns}"
    if arm == "swapped":
        turns = _turns_with_answers(
            row["report_queries"],
            row["swapped_report_tokens"],
            title="A report gives two consequence checks.",
        )
        return f"{scenario}\n\n{turns}"
    if arm == "matched_control":
        turns = _turns_with_answers(
            row["control_queries"],
            [query["expected_token"] for query in row["control_queries"]],
            title="Before acting, answer two neutral control checks.",
        )
        return f"{scenario}\n\n{turns}"
    if arm == "external_facts":
        turns = _turns_with_answers(
            row["report_queries"],
            row["expected_report_tokens"],
            title="An external reference card supplies two consequence checks.",
        )
        return f"{scenario}\n\n{turns}"
    raise ValueError(f"unknown primary arm: {arm}")


def _with_final_action_query(prefix: str, *, use_reports_instruction: bool = False) -> str:
    if use_reports_instruction:
        prefix = f"{prefix}\nUse the reports as evidence when selecting the action."
    return f"{prefix}\n\n{COMMON_FINAL_ACTION_QUERY_SUFFIX}"


def _row_for_arm(variant: Mapping[str, Any], arm: str) -> dict[str, Any]:
    prefix = _arm_prefix(variant, arm)
    if arm in {"oracle", "external_facts"}:
        arm_report_tokens: list[str] | None = list(variant["expected_report_tokens"])
    elif arm == "swapped":
        arm_report_tokens = list(variant["swapped_report_tokens"])
    elif arm == "matched_control":
        arm_report_tokens = [query["expected_token"] for query in variant["control_queries"]]
    else:
        arm_report_tokens = None
    condition_id = canonical_sha256(
        [
            variant["base_game_id"],
            variant["frame"],
            variant["surface_kind"],
            variant["policy_kind"],
            arm,
        ]
    )[:16]
    return {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "condition_id": condition_id,
        "base_game_id": variant["base_game_id"],
        "lexical_group_id": variant["lexical_group_id"],
        "split": variant["split"],
        "base_index": variant["base_index"],
        "frame": variant["frame"],
        "surface_kind": variant["surface_kind"],
        "policy_kind": variant["policy_kind"],
        "history_mode": variant["history_mode"],
        "arm": arm,
        "concepts": list(variant["concepts"]),
        "opaque_tokens": dict(variant["opaque_tokens"]),
        "action_text": dict(variant["action_text"]),
        "action_to_outcome": dict(variant["action_to_outcome"]),
        "target_outcome": variant["target_outcome"],
        "expected_action": variant["expected_action"],
        "action_order": list(variant["action_order"]),
        "target_index": variant["target_index"],
        "outcome_order": list(variant["outcome_order"]),
        "report_order": list(variant["report_order"]),
        "query_order": list(variant["query_order"]),
        "demonstrations": [dict(item) for item in variant["demonstrations"]],
        "report_mapping": dict(variant["report_mapping"]),
        "report_queries": [dict(query) for query in variant["report_queries"]],
        "control_queries": [dict(query) for query in variant["control_queries"]],
        "expected_report_tokens": list(variant["expected_report_tokens"]),
        "swapped_report_tokens": list(variant["swapped_report_tokens"]),
        "arm_report_tokens": arm_report_tokens,
        "oracle": dict(variant["oracle"]),
        "scenario": variant["scenario"],
        "prefix": prefix,
        "final_action_query": COMMON_FINAL_ACTION_QUERY_SUFFIX,
        "prompt": _with_final_action_query(prefix),
        "primary": True,
    }


def generate_rows(
    *,
    discovery_bases: int = 48,
    locked_bases: int = 96,
    history_mode: str = DEFAULT_HISTORY_MODE,
) -> list[dict[str, Any]]:
    """Generate rows for all split, frame, surface, and primary-arm cells."""

    if discovery_bases < 0 or locked_bases < 0 or discovery_bases + locked_bases == 0:
        raise ValueError("at least one base is required")
    history_mode = _normalise_history_mode(history_mode)
    rows: list[dict[str, Any]] = []
    for split, count in (("discovery", discovery_bases), ("locked", locked_bases)):
        for base_index in range(count):
            base = _base_metadata(split, base_index)
            for frame in FRAMES:
                for surface_kind in SURFACE_KINDS:
                    for policy_kind in POLICY_KINDS:
                        variant = _make_variant(
                            base,
                            frame=frame,
                            surface_kind=surface_kind,
                            policy_kind=policy_kind,
                            history_mode=history_mode,
                        )
                        variant["scenario"] = _render_scenario(
                            frame=frame,
                            surface_kind=surface_kind,
                            concepts=variant["concepts"],
                            action_to_outcome=variant["action_to_outcome"],
                            action_text=variant["action_text"],
                            target=variant["target_outcome"],
                            demonstrations=variant["demonstrations"],
                        )
                        rows.extend(_row_for_arm(variant, arm) for arm in PRIMARY_ARMS)
    return rows


def generate_corpus(
    *,
    discovery_bases: int = 48,
    locked_bases: int = 96,
    history_mode: str = DEFAULT_HISTORY_MODE,
) -> dict[str, Any]:
    """Generate a hashable payload using the proposed discovery/locked sizes."""

    config = _normalise_config(
        {
            "history_mode": history_mode,
            "dataset": {
                "discovery_bases": discovery_bases,
                "locked_bases": locked_bases,
            },
        }
    )
    body = {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "config_sha256": canonical_sha256(config),
        "rows": generate_rows(
            discovery_bases=discovery_bases,
            locked_bases=locked_bases,
            history_mode=config["history_mode"],
        ),
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def dataset_payload(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Generate a payload from a config in the same shape as other game modules."""

    normalised = _normalise_config(config)
    dataset = normalised["dataset"]
    body = {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "config_sha256": canonical_sha256(normalised),
        "rows": generate_rows(
            discovery_bases=dataset["discovery_bases"],
            locked_bases=dataset["locked_bases"],
            history_mode=normalised["history_mode"],
        ),
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def _row_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["split"],
        row["base_game_id"],
        row["frame"],
        row["surface_kind"],
        row["policy_kind"],
    )


def verify_dataset_payload(
    payload: Mapping[str, Any], config: Mapping[str, Any] | None = None
) -> None:
    """Validate hashes, factorial support, balancing metadata, and oracle facts."""

    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    if payload.get("content_sha256") != canonical_sha256(body):
        raise ValueError("dataset content hash mismatch")
    normalised = _normalise_config(config)
    if payload.get("config_sha256") != canonical_sha256(normalised):
        raise ValueError("dataset/config hash mismatch")
    rows = list(payload.get("rows", []))
    expected = (
        (normalised["dataset"]["discovery_bases"] + normalised["dataset"]["locked_bases"])
        * len(FRAMES)
        * len(SURFACE_KINDS)
        * len(POLICY_KINDS)
        * len(PRIMARY_ARMS)
    )
    if len(rows) != expected:
        raise ValueError(f"unexpected factorial size: {len(rows)} != {expected}")
    ids = [row.get("condition_id") for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("condition IDs are not unique")
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    expected_demo_count = 2 * _demo_repeats(normalised["history_mode"])
    for row in rows:
        grouped.setdefault(_row_key(row), []).append(row)
        if row.get("arm") not in PRIMARY_ARMS:
            raise ValueError("unknown primary arm")
        if row.get("history_mode") != normalised["history_mode"]:
            raise ValueError("row history_mode does not match config")
        if len(row.get("demonstrations", [])) != expected_demo_count:
            raise ValueError(
                f"unexpected demonstration count: "
                f"{len(row.get('demonstrations', []))} != {expected_demo_count}"
            )
        if not str(row.get("prompt", "")).endswith(COMMON_FINAL_ACTION_QUERY_SUFFIX):
            raise ValueError("primary prompt does not end in the common action query")
        oracle = row.get("oracle", {})
        action = oracle.get("expected_action")
        if action != row.get("expected_action"):
            raise ValueError("oracle/action mismatch")
        if row.get("action_to_outcome", {}).get(action) != row.get("target_outcome"):
            raise ValueError("expected action does not achieve target")
        expected_reports = [query["expected_token"] for query in row["report_queries"]]
        if expected_reports != row.get("expected_report_tokens"):
            raise ValueError("report oracle mismatch")
        if len(row.get("report_queries", [])) != 2 or len(row.get("control_queries", [])) != 2:
            raise ValueError("each arm must carry two report and two control queries")
    for key, cell in grouped.items():
        arms = {row["arm"] for row in cell}
        if arms != set(PRIMARY_ARMS):
            raise ValueError(f"primary arms incomplete for {key}")
        scenarios = {row["scenario"] for row in cell}
        if len(scenarios) != 1:
            raise ValueError(f"primary arms do not share one scenario for {key}")


def render_self_report_prompt(
    row: Mapping[str, Any],
    generated_tokens: Sequence[str],
    *,
    use_reports_instruction: bool = False,
) -> str:
    """Render the action prompt after inserting the model's two actual tokens."""

    _validate_generated_tokens(generated_tokens)
    prefix = _arm_prefix(row, "self_report", generated_tokens)
    return _with_final_action_query(prefix, use_reports_instruction=use_reports_instruction)


def _validate_generated_tokens(generated_tokens: Sequence[str]) -> None:
    if isinstance(generated_tokens, (str, bytes)):
        raise TypeError("generated_tokens must be a sequence of the two token strings")
    if len(generated_tokens) != 2:
        raise ValueError("exactly two generated report tokens are required")
    if any(not isinstance(token, str) for token in generated_tokens):
        raise TypeError("generated report tokens must be strings")


def self_report_branch(
    row: Mapping[str, Any],
    generated_tokens: Sequence[str],
    *,
    use_reports_instruction: bool = False,
) -> dict[str, Any]:
    """Build a serializable action branch from the two actual generated reports."""

    _validate_generated_tokens(generated_tokens)
    tokens = list(generated_tokens)
    expected = list(row["expected_report_tokens"])
    return {
        "schema_version": SCHEMA_VERSION,
        "branch_id": "self_report_action",
        "source_condition_id": row["condition_id"],
        "base_game_id": row["base_game_id"],
        "split": row["split"],
        "shared_prefix_hash": canonical_sha256(row["scenario"]),
        "generated_tokens": tokens,
        "expected_tokens": expected,
        "report_correct": [
            token.strip() == target for token, target in zip(tokens, expected, strict=True)
        ],
        "both_reports_correct": all(
            token.strip() == target for token, target in zip(tokens, expected, strict=True)
        ),
        "use_reports_instruction": use_reports_instruction,
        "prompt": render_self_report_prompt(
            row, tokens, use_reports_instruction=use_reports_instruction
        ),
        "oracle": dict(row["oracle"]),
    }


def _validate_prior_reports(prior_reports: Sequence[str]) -> list[str]:
    if isinstance(prior_reports, (str, bytes)):
        raise TypeError("prior_reports must be a sequence of report strings")
    if len(prior_reports) > 2:
        raise ValueError("at most two prior reports are supported")
    if any(not isinstance(token, str) for token in prior_reports):
        raise TypeError("prior reports must be strings")
    return list(prior_reports)


def report_messages(
    row: Mapping[str, Any], prior_reports: Sequence[str] = ()
) -> list[dict[str, str]]:
    """Return the next independent report turn from a scenario prefix.

    The returned messages are suitable for sequential generation: prior model
    outputs are assistant turns, and the next consequence question is a user
    turn.  With two prior reports the list ends after the second assistant turn
    and is ready for the common final action request to be appended.
    """

    reports = _validate_prior_reports(prior_reports)
    messages: list[dict[str, str]] = [{"role": "user", "content": str(row["scenario"])}]
    for index, token in enumerate(reports):
        query = row["report_queries"][index]
        messages.append({"role": "user", "content": query["question"]})
        messages.append({"role": "assistant", "content": token})
    if len(reports) < 2:
        messages.append(
            {"role": "user", "content": row["report_queries"][len(reports)]["question"]}
        )
    return messages


def arm_messages(
    row: Mapping[str, Any], reports: Sequence[str] | None = None
) -> list[dict[str, str]]:
    """Render an arm as chat messages with a common final user request.

    For a self-report row, ``reports`` are the actual generated assistant
    tokens.  Omitting them emits explicit placeholders so a CPU runner can
    inspect message structure; it must replace those placeholders before model
    scoring.  Oracle, swapped, matched-control, and external-facts rows carry
    their frozen answers in ``arm_report_tokens``.
    """

    arm = str(row["arm"])
    if arm == "self_report":
        if reports is None:
            answer_tokens = ["<GENERATED_REPORT_0>", "<GENERATED_REPORT_1>"]
        else:
            _validate_generated_tokens(reports)
            answer_tokens = list(reports)
    else:
        if reports is not None:
            raise ValueError("reports can only be supplied for the self_report arm")
        answer_tokens = row.get("arm_report_tokens")

    messages: list[dict[str, str]] = [{"role": "user", "content": str(row["scenario"])}]
    if arm == "direct":
        pass
    elif arm == "external_facts":
        messages.append(
            {
                "role": "user",
                "content": (
                    "An external reference card supplies the following consequence checks."
                ),
            }
        )
    queries = row["control_queries"] if arm == "matched_control" else row["report_queries"]
    if answer_tokens is not None:
        for query, token in zip(queries, answer_tokens, strict=True):
            messages.append({"role": "user", "content": query["question"]})
            messages.append({"role": "assistant", "content": str(token)})
    elif arm == "self_report":
        # This branch is a structural template only; use report_messages or
        # self_report_branch once the model has actually generated both tokens.
        for index, query in enumerate(queries):
            messages.append({"role": "user", "content": query["question"]})
            messages.append({"role": "assistant", "content": f"<GENERATED_REPORT_{index}>"})
    messages.append({"role": "user", "content": COMMON_FINAL_ACTION_QUERY_SUFFIX})
    return messages


def independent_fork_queries(row: Mapping[str, Any]) -> dict[str, Any]:
    """Create separately recomputed action/report branches from one prefix.

    The record describes the capture points a model runner should use.  It does
    not claim that a Python string copy is a valid Transformer KV-cache copy.
    Every branch therefore carries ``recompute_full_prefix=True``.
    """

    shared_prefix = str(row["scenario"])
    shared_hash = canonical_sha256(shared_prefix)
    branches: list[dict[str, Any]] = [
        {
            "branch_id": "direct_action",
            "branch_kind": "action",
            "parent_branch": "shared_prefix",
            "query": COMMON_FINAL_ACTION_QUERY_SUFFIX,
            "prompt": _with_final_action_query(shared_prefix),
            "recompute_full_prefix": True,
            "oracle": {"expected_action": row["expected_action"]},
            "messages": arm_messages({**dict(row), "arm": "direct", "arm_report_tokens": None}),
        }
    ]
    for query in row["report_queries"]:
        branch_id = f"report_{query['query_index']}"
        branches.append(
            {
                "branch_id": branch_id,
                "branch_kind": "report",
                "parent_branch": "shared_prefix",
                "query": query["question"] + "\nAnswer:",
                "prompt": f"{shared_prefix}\n\n{query['question']}\nAnswer:",
                "recompute_full_prefix": True,
                "oracle": {
                    "expected_token": query["expected_token"],
                    "expected_outcome": query["expected_outcome"],
                    "action": query["action"],
                },
                "messages": [
                    {"role": "user", "content": shared_prefix},
                    {"role": "user", "content": query["question"]},
                ],
            }
        )
    for branch in branches:
        branch["shared_prefix_hash"] = shared_hash
        branch["capture_points"] = [
            {"name": "shared_prefix", "state_role": "pre_query", "post_query": False},
            {"name": "final_pre_answer", "state_role": "post_query", "post_query": True},
        ]
    return {
        "schema_version": SCHEMA_VERSION,
        "source_condition_id": row["condition_id"],
        "base_game_id": row["base_game_id"],
        "shared_prefix": shared_prefix,
        "shared_prefix_hash": shared_hash,
        "branches": branches,
    }


def _mid_turn_spec(variant: Mapping[str, Any], arm: str) -> dict[str, Any]:
    """Return the mid-trajectory question and frozen answer (if any).

    Ask arms use the first consequence check; the no-ask control uses the first
    matched formatting check so length and forced-choice shape stay comparable.
    """

    if arm == "mid_no_ask_control":
        query = dict(variant["control_queries"][0])
        return {
            "mid_kind": "control",
            "mid_query": query,
            "expected_mid_token": query["expected_token"],
            "arm_mid_token": query["expected_token"],
            "swapped_mid_token": REPORT_LABELS[
                1 - REPORT_LABELS.index(query["expected_token"])
            ],
        }
    query = dict(variant["report_queries"][0])
    expected = query["expected_token"]
    swapped = REPORT_LABELS[1 - REPORT_LABELS.index(expected)]
    if arm == "mid_ask_self":
        arm_mid_token: str | None = None
    elif arm == "mid_ask_oracle":
        arm_mid_token = expected
    elif arm == "mid_ask_swapped":
        arm_mid_token = swapped
    else:
        raise ValueError(f"unknown mid-trajectory arm: {arm}")
    return {
        "mid_kind": "consequence",
        "mid_query": query,
        "expected_mid_token": expected,
        "arm_mid_token": arm_mid_token,
        "swapped_mid_token": swapped,
    }


def _row_for_mid_arm(variant: Mapping[str, Any], arm: str) -> dict[str, Any]:
    if arm not in MID_TRAJECTORY_ARMS:
        raise ValueError(f"unknown mid-trajectory arm: {arm}")
    mid = _mid_turn_spec(variant, arm)
    condition_id = canonical_sha256(
        [
            MID_TRAJECTORY_STUDY_ID,
            variant["base_game_id"],
            variant["frame"],
            variant["surface_kind"],
            variant["policy_kind"],
            arm,
        ]
    )[:16]
    return {
        "schema_version": SCHEMA_VERSION,
        "study_id": MID_TRAJECTORY_STUDY_ID,
        "protocol": "ask_mid_trajectory",
        "condition_id": condition_id,
        "base_game_id": variant["base_game_id"],
        "lexical_group_id": variant["lexical_group_id"],
        "split": variant["split"],
        "base_index": variant["base_index"],
        "frame": variant["frame"],
        "surface_kind": variant["surface_kind"],
        "policy_kind": variant["policy_kind"],
        "history_mode": variant["history_mode"],
        "arm": arm,
        "concepts": list(variant["concepts"]),
        "opaque_tokens": dict(variant["opaque_tokens"]),
        "action_text": dict(variant["action_text"]),
        "action_to_outcome": dict(variant["action_to_outcome"]),
        "target_outcome": variant["target_outcome"],
        "expected_action": variant["expected_action"],
        "action_order": list(variant["action_order"]),
        "target_index": variant["target_index"],
        "outcome_order": list(variant["outcome_order"]),
        "report_order": list(variant["report_order"]),
        "query_order": list(variant["query_order"]),
        "demonstrations": [dict(item) for item in variant["demonstrations"]],
        "report_mapping": dict(variant["report_mapping"]),
        "report_queries": [dict(query) for query in variant["report_queries"]],
        "control_queries": [dict(query) for query in variant["control_queries"]],
        "expected_report_tokens": list(variant["expected_report_tokens"]),
        "swapped_report_tokens": list(variant["swapped_report_tokens"]),
        "oracle": dict(variant["oracle"]),
        "scenario": variant["scenario"],
        "mid_kind": mid["mid_kind"],
        "mid_query": dict(mid["mid_query"]),
        "expected_mid_token": mid["expected_mid_token"],
        "arm_mid_token": mid["arm_mid_token"],
        "swapped_mid_token": mid["swapped_mid_token"],
        "final_action_query": COMMON_FINAL_ACTION_QUERY_SUFFIX,
        "primary": True,
    }


def generate_mid_trajectory_rows(
    *,
    discovery_bases: int = 48,
    locked_bases: int = 0,
    history_mode: str = DEFAULT_HISTORY_MODE,
) -> list[dict[str, Any]]:
    """Generate ask-mid-trajectory rows (choice → mid ask/control → choice).

    Reuses the same scenario family as the six-arm report corpus. Locked bases
    default to zero because this protocol is discovery-first and not yet scored.
    """

    if discovery_bases < 0 or locked_bases < 0 or discovery_bases + locked_bases == 0:
        raise ValueError("at least one base is required")
    history_mode = _normalise_history_mode(history_mode)
    rows: list[dict[str, Any]] = []
    for split, count in (("discovery", discovery_bases), ("locked", locked_bases)):
        for base_index in range(count):
            base = _base_metadata(split, base_index)
            for frame in FRAMES:
                for surface_kind in SURFACE_KINDS:
                    for policy_kind in POLICY_KINDS:
                        variant = _make_variant(
                            base,
                            frame=frame,
                            surface_kind=surface_kind,
                            policy_kind=policy_kind,
                            history_mode=history_mode,
                        )
                        variant["scenario"] = _render_scenario(
                            frame=frame,
                            surface_kind=surface_kind,
                            concepts=variant["concepts"],
                            action_to_outcome=variant["action_to_outcome"],
                            action_text=variant["action_text"],
                            target=variant["target_outcome"],
                            demonstrations=variant["demonstrations"],
                        )
                        rows.extend(
                            _row_for_mid_arm(variant, arm) for arm in MID_TRAJECTORY_ARMS
                        )
    return rows


def mid_trajectory_dataset_payload(
    *,
    discovery_bases: int = 16,
    locked_bases: int = 0,
    history_mode: str = DEFAULT_HISTORY_MODE,
) -> dict[str, Any]:
    """Hashable mid-trajectory corpus payload (CPU-side, no tokenizer)."""

    history_mode = _normalise_history_mode(history_mode)
    config = {
        "schema_version": SCHEMA_VERSION,
        "study_id": MID_TRAJECTORY_STUDY_ID,
        "protocol": "ask_mid_trajectory",
        "history_mode": history_mode,
        "dataset": {
            "discovery_bases": discovery_bases,
            "locked_bases": locked_bases,
            "frames": list(FRAMES),
            "surface_kinds": list(SURFACE_KINDS),
            "policy_kinds": list(POLICY_KINDS),
            "mid_trajectory_arms": list(MID_TRAJECTORY_ARMS),
        },
    }
    body = {
        "schema_version": SCHEMA_VERSION,
        "study_id": MID_TRAJECTORY_STUDY_ID,
        "protocol": "ask_mid_trajectory",
        "config_sha256": canonical_sha256(config),
        "rows": generate_mid_trajectory_rows(
            discovery_bases=discovery_bases,
            locked_bases=locked_bases,
            history_mode=history_mode,
        ),
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def verify_mid_trajectory_payload(payload: Mapping[str, Any]) -> None:
    """Validate mid-trajectory factorial support and shared scenarios."""

    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    if payload.get("content_sha256") != canonical_sha256(body):
        raise ValueError("mid-trajectory content hash mismatch")
    if payload.get("protocol") != "ask_mid_trajectory":
        raise ValueError("payload protocol must be ask_mid_trajectory")
    if payload.get("study_id") != MID_TRAJECTORY_STUDY_ID:
        raise ValueError("unexpected mid-trajectory study_id")
    rows = list(payload.get("rows", []))
    if not rows:
        raise ValueError("mid-trajectory payload has no rows")
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("arm") not in MID_TRAJECTORY_ARMS:
            raise ValueError(f"unknown mid-trajectory arm: {row.get('arm')}")
        if row.get("protocol") != "ask_mid_trajectory":
            raise ValueError("row protocol mismatch")
        if not row.get("mid_query"):
            raise ValueError("mid_query required")
        if row["arm"] == "mid_ask_self" and row.get("arm_mid_token") is not None:
            raise ValueError("mid_ask_self must not freeze arm_mid_token")
        if row["arm"] != "mid_ask_self" and row.get("arm_mid_token") is None:
            raise ValueError(f"{row['arm']} requires frozen arm_mid_token")
        grouped.setdefault(_row_key(row), []).append(row)
    for key, cell in grouped.items():
        arms = {row["arm"] for row in cell}
        if arms != set(MID_TRAJECTORY_ARMS):
            raise ValueError(f"mid-trajectory arms incomplete for {key}")
        scenarios = {row["scenario"] for row in cell}
        if len(scenarios) != 1:
            raise ValueError(f"mid-trajectory arms do not share one scenario for {key}")
        ask = next(row for row in cell if row["arm"] == "mid_ask_self")
        control = next(row for row in cell if row["arm"] == "mid_no_ask_control")
        if ask["mid_query"]["question"] == control["mid_query"]["question"]:
            raise ValueError("ask and control mid questions must differ")
        if ask["mid_kind"] != "consequence" or control["mid_kind"] != "control":
            raise ValueError("mid_kind mismatch for ask vs control")


def mid_choice1_messages(row: Mapping[str, Any]) -> list[dict[str, str]]:
    """First forced action choice (Direct-style final instruction)."""

    return [
        {"role": "user", "content": str(row["scenario"])},
        {"role": "user", "content": COMMON_FINAL_ACTION_QUERY_SUFFIX},
    ]


def mid_turn_messages(
    row: Mapping[str, Any], choice1: str, *, answer: str | None = None
) -> list[dict[str, str]]:
    """Messages through the mid ask/control turn after ``choice1``.

    When ``answer`` is omitted, ends on the mid user question (for scoring the
    model's mid response). When supplied, appends the assistant mid answer.
    """

    if choice1 not in ACTION_LABELS:
        raise ValueError(f"choice1 must be one of {ACTION_LABELS}")
    messages = mid_choice1_messages(row) + [
        {"role": "assistant", "content": choice1},
        {"role": "user", "content": str(row["mid_query"]["question"])},
    ]
    if answer is not None:
        if not isinstance(answer, str):
            raise TypeError("mid answer must be a string")
        messages.append({"role": "assistant", "content": answer})
    return messages


def mid_choice2_messages(
    row: Mapping[str, Any], choice1: str, mid_answer: str
) -> list[dict[str, str]]:
    """Second forced action choice after the mid turn (same final instruction)."""

    if mid_answer not in REPORT_LABELS:
        raise ValueError(f"mid_answer must be one of {REPORT_LABELS}")
    return mid_turn_messages(row, choice1, answer=mid_answer) + [
        {"role": "user", "content": COMMON_FINAL_ACTION_QUERY_SUFFIX},
    ]


def mid_ask_vs_control_prefix_diff(row_ask: Mapping[str, Any], row_control: Mapping[str, Any]) -> dict[str, Any]:
    """Assert ask vs control share scenario and differ only in the mid question."""

    if row_ask["scenario"] != row_control["scenario"]:
        raise ValueError("ask/control scenarios differ")
    if row_ask["base_game_id"] != row_control["base_game_id"]:
        raise ValueError("ask/control base_game_id differ")
    ask_q = str(row_ask["mid_query"]["question"])
    ctrl_q = str(row_control["mid_query"]["question"])
    return {
        "scenario_equal": True,
        "choice1_instruction_equal": True,
        "choice2_instruction_equal": True,
        "mid_questions_differ": ask_q != ctrl_q,
        "ask_mid_kind": row_ask["mid_kind"],
        "control_mid_kind": row_control["mid_kind"],
        "ask_mid_question": ask_q,
        "control_mid_question": ctrl_q,
    }


# Descriptive aliases make the API discoverable for runners without duplicating
# the implementation.
build_self_report_branch = self_report_branch
build_fork_queries = independent_fork_queries
fork_report_queries = independent_fork_queries
corpus_payload = dataset_payload


__all__ = [
    "ACTION_LABELS",
    "COMMON_FINAL_ACTION_QUERY",
    "COMMON_FINAL_ACTION_QUERY_SUFFIX",
    "DEFAULT_CONFIG",
    "DEFAULT_HISTORY_MODE",
    "FRAMES",
    "HISTORY_MODES",
    "MID_TRAJECTORY_ARMS",
    "MID_TRAJECTORY_LEGACY_STUDY_ID",
    "MID_TRAJECTORY_STUDY_ID",
    "MINIMAL_DEMO_REPEATS",
    "PRIMARY_ARMS",
    "REDUNDANT_EXTRA_CYCLES",
    "REPORT_LABELS",
    "SPLITS",
    "SURFACE_KINDS",
    "STUDY_ID",
    "build_fork_queries",
    "build_self_report_branch",
    "canonical_sha256",
    "corpus_payload",
    "dataset_payload",
    "fork_report_queries",
    "generate_corpus",
    "generate_mid_trajectory_rows",
    "generate_rows",
    "independent_fork_queries",
    "mid_ask_vs_control_prefix_diff",
    "mid_choice1_messages",
    "mid_choice2_messages",
    "mid_trajectory_dataset_payload",
    "mid_turn_messages",
    "render_self_report_prompt",
    "self_report_branch",
    "verify_dataset_payload",
    "verify_mid_trajectory_payload",
]
