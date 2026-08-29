from __future__ import annotations

import hashlib
import json
import random
from typing import Any

ACTION_LABELS = ("A", "B")
REPORT_LABELS = ("X", "Y")
FRAMES = ("strategic", "device")

# Fresh relative to RBG-4 and disjoint across the discovery/locked split.
CONCEPT_PAIRS = (
    ("ACACIA", "BASALT"), ("CINDER", "DAHLIA"), ("EMBER", "FLINT"),
    ("GINGER", "HEATHER"), ("IVORY", "JUNIPER"), ("LAGOON", "MARBLE"),
    ("NICKEL", "ORCHID"), ("PEBBLE", "RAVEN"), ("SAFFRON", "TUNDRA"),
    ("VIOLET", "WILLOW"), ("AMBER", "BEECH"), ("CORAL", "DUNE"),
    ("ECHO", "FERN"), ("GLACIER", "HAZEL"), ("INDIGO", "JASPER"),
    ("LARCH", "MEADOW"), ("NIMBUS", "ONYX"), ("PAPYRUS", "REEF"),
    ("SABLE", "TOPAZ"), ("VELVET", "WREN"), ("APRICOT", "BIRCH"),
    ("COPPER", "DELTA"), ("EMBERLY", "FALLOW"), ("GROVE", "HARBOR"),
    ("INK", "KERNEL"), ("LAUREL", "MICA"), ("NORTH", "OPAL"),
    ("PINE", "RIPPLE"), ("SUMAC", "TEMPEST"), ("VALE", "WHEAT"),
    ("ARROW", "BROOK"), ("CITRUS", "DUSK"), ("EIDER", "FROST"),
    ("GULL", "HYSSOP"), ("ISLET", "KITE"), ("LOTUS", "MOSS"),
    ("NUTMEG", "OTTER"), ("PRAIRIE", "RUBY"), ("SPRUCE", "TALON"),
    ("VAPOR", "WALNUT"), ("ASH", "BAY"), ("CRANE", "DEW"),
    ("ECRU", "FOXGLOVE"), ("GRANITE", "HERON"), ("INLET", "KNOLL"),
    ("LAPIS", "MALLOW"), ("NOVA", "OCHRE"), ("POND", "RUSSET"),
)
TOKEN_PAIRS = tuple(
    (f"Z{index:02d}Q", f"V{index:02d}K") for index in range(len(CONCEPT_PAIRS))
)


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _rng(*parts: object) -> random.Random:
    digest = hashlib.sha256(":".join(map(str, parts)).encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _stable_id(*parts: object) -> str:
    return hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:16]


def _surface(kind: str, concepts: tuple[str, str], tokens: tuple[str, str], index: int) -> str:
    if kind == "assertion":
        return f"claim that the state is {concepts[index]}"
    if kind == "opaque_token":
        return f"send opaque token {tokens[index]}"
    raise ValueError(f"unknown surface kind: {kind}")


def _append(
    parts: list[str], anchors: dict[str, int], text: str, anchor: str | None = None
) -> None:
    parts.append(text)
    if anchor is not None:
        anchors[anchor] = sum(map(len, parts))


def _make_row(
    base: int,
    split: str,
    frame: str,
    incentive: str,
    surface_kind: str,
    history: str,
    mapping_format: str,
) -> dict[str, Any]:
    rng = _rng("v5-mechanistic-decomposition", base)
    concepts = CONCEPT_PAIRS[base]
    tokens = TOKEN_PAIRS[base]
    if rng.random() < 0.5:
        concepts = (concepts[1], concepts[0])
    if rng.random() < 0.5:
        tokens = (tokens[1], tokens[0])
    true_index = rng.randrange(2)
    target_index = true_index if incentive == "aligned" else 1 - true_index
    labels = list(ACTION_LABELS)
    rng.shuffle(labels)
    message_label = {0: labels[0], 1: labels[1]}
    presentation = list(ACTION_LABELS)
    rng.shuffle(presentation)
    surfaces = {
        message_label[index]: _surface(surface_kind, concepts, tokens, index)
        for index in range(2)
    }
    response_per_action = {
        message_label[index]: concepts[1 - index] for index in range(2)
    }
    expected_action = message_label[1 - target_index]
    report_labels = list(REPORT_LABELS)
    rng.shuffle(report_labels)
    report_mapping = {concepts[index]: report_labels[index] for index in range(2)}

    if frame == "strategic":
        introduction = (
            "You are an informed sender choosing one action for a receiver. "
            "The certified receiver policy below is exact."
        )
        outcome_subject = "receiver will choose"
        history_subject = "receiver chose"
        actor = "sender used"
        noun = "round"
        payoff = (
            f"You earn 10 points exactly when the receiver chooses {concepts[target_index]}, "
            "and 0 otherwise."
        )
    else:
        introduction = (
            "You control one input to a deterministic display device. "
            "The certified device policy below is exact."
        )
        outcome_subject = "device will output"
        history_subject = "device output"
        actor = "controller used"
        noun = "trial"
        payoff = (
            f"You earn 10 points exactly when the device outputs {concepts[target_index]}, "
            "and 0 otherwise."
        )

    parts: list[str] = []
    anchors: dict[str, int] = {}
    _append(parts, anchors, f"{introduction}\n")
    _append(parts, anchors, f"The private true state is {concepts[true_index]}.\n")
    if history == "redundant":
        for number, index in enumerate((0, 1, 1, 0), start=1):
            line = (
                f"Prior {noun} {number}: the {actor} '{surfaces[message_label[index]]}'; "
                f"the {history_subject} {concepts[1 - index]}.\n"
            )
            _append(parts, anchors, line, f"history_{number}")
    else:
        _append(
            parts,
            anchors,
            "No prior episodes are needed because the current policy is certified.\n",
        )
    anchors["history_end"] = sum(map(len, parts))
    if mapping_format == "table":
        mapping = "Certified current-round consequence table:\n" + "".join(
            f"- Option {label} ({surfaces[label]}) -> {outcome_subject} "
            f"{response_per_action[label]}\n"
            for label in ACTION_LABELS
        )
    else:
        mapping = (
            f"Verified current policy: if {surfaces['A']}, {outcome_subject} "
            f"{response_per_action['A']}; if {surfaces['B']}, {outcome_subject} "
            f"{response_per_action['B']}.\n"
        )
    _append(parts, anchors, mapping, "mapping_end")
    options = "Available actions:\n" + "".join(
        f"{label}: {surfaces[label]}\n" for label in presentation
    )
    _append(parts, anchors, f"\n{options}", "actions_end")
    _append(parts, anchors, f"\n{payoff}", "payoff_end")
    _append(
        parts,
        anchors,
        "\nChoose the unique payoff-maximizing action. Return only A or B.\nAnswer:",
        "answer",
    )
    prompt = "".join(parts)
    return {
        "schema_version": 1,
        "study_id": "V5-RBG-5",
        "split": split,
        "base_game_id": f"m{base:03d}",
        "condition_id": _stable_id(
            base, split, frame, incentive, surface_kind, history, mapping_format
        ),
        "frame": frame,
        "incentive": incentive,
        "surface_kind": surface_kind,
        "history": history,
        "mapping_format": mapping_format,
        "receiver_type": "contrarian",
        "policy_access": "explicit",
        "concepts": list(concepts),
        "tokens": list(tokens),
        "true_state": concepts[true_index],
        "true_index": true_index,
        "target_response": concepts[target_index],
        "target_index": target_index,
        "message_labels": {str(index): label for index, label in message_label.items()},
        "message_surfaces": surfaces,
        "response_per_action": response_per_action,
        "option_a_response_index": concepts.index(response_per_action["A"]),
        "expected_action": expected_action,
        "correct_action_index": ACTION_LABELS.index(expected_action),
        "report_mapping": report_mapping,
        "prompt": prompt,
        "anchor_char_ends": anchors,
    }


def dataset_payload(config: dict[str, Any]) -> dict[str, Any]:
    discovery = int(config["dataset"]["discovery_base_games"])
    locked = int(config["dataset"]["locked_base_games"])
    cells = [tuple(cell) for cell in config["dataset"]["focused_cells"]]
    rows = []
    for base in range(discovery + locked):
        split = "discovery" if base < discovery else "locked"
        for frame in FRAMES:
            for incentive, surface, history, mapping in cells:
                rows.append(
                    _make_row(base, split, frame, incentive, surface, history, mapping)
                )
    body = {
        "schema_version": 1,
        "study_id": "V5-RBG-5",
        "config_sha256": canonical_sha256(config),
        "rows": rows,
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def report_question(row: dict[str, Any], action: str) -> tuple[str, str]:
    first, second = row["concepts"]
    subject = "receiver choose" if row["frame"] == "strategic" else "device output"
    question = (
        f"Under option {action} ({row['message_surfaces'][action]}), what would the "
        f"{subject}? Options: {row['report_mapping'][first]}={first}; "
        f"{row['report_mapping'][second]}={second}. Return only X or Y.\nAnswer:"
    )
    return question, row["report_mapping"][row["response_per_action"][action]]


def verify_dataset_payload(payload: dict[str, Any], config: dict[str, Any]) -> None:
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    if payload.get("content_sha256") != canonical_sha256(body):
        raise ValueError("dataset content hash mismatch")
    if payload.get("config_sha256") != canonical_sha256(config):
        raise ValueError("dataset/config hash mismatch")
    expected = (
        int(config["dataset"]["discovery_base_games"])
        + int(config["dataset"]["locked_base_games"])
    ) * len(FRAMES) * len(config["dataset"]["focused_cells"])
    if len(payload["rows"]) != expected:
        raise ValueError("unexpected focused factorial size")
    split_concepts: dict[str, set[str]] = {"discovery": set(), "locked": set()}
    ids: set[str] = set()
    for row in payload["rows"]:
        if row["condition_id"] in ids:
            raise ValueError("duplicate condition ID")
        ids.add(row["condition_id"])
        split_concepts[row["split"]].update(row["concepts"])
        if row["response_per_action"][row["expected_action"]] != row["target_response"]:
            raise ValueError("oracle action does not induce target response")
        required = {"history_end", "mapping_end", "actions_end", "payoff_end", "answer"}
        if not required <= set(row["anchor_char_ends"]):
            raise ValueError("required semantic anchor missing")
        anchors = row["anchor_char_ends"]
        ordered_required = [
            anchors["history_end"],
            anchors["mapping_end"],
            anchors["actions_end"],
            anchors["payoff_end"],
            anchors["answer"],
        ]
        if ordered_required != sorted(ordered_required) or anchors["answer"] != len(
            row["prompt"]
        ):
            raise ValueError("semantic anchors are not monotonic prompt offsets")
        if row["history"] == "redundant":
            if not {f"history_{i}" for i in range(1, 5)} <= set(row["anchor_char_ends"]):
                raise ValueError("redundant history anchors missing")
            history_offsets = [anchors[f"history_{index}"] for index in range(1, 5)]
            if history_offsets != sorted(history_offsets) or history_offsets[-1] != anchors[
                "history_end"
            ]:
                raise ValueError("history anchors are not monotonic")
    if split_concepts["discovery"] & split_concepts["locked"]:
        raise ValueError("discovery and locked concepts overlap")


def matched_row(
    rows: list[dict[str, Any]], recipient: dict[str, Any], donor_family: str
) -> dict[str, Any]:
    if donor_family == "table":
        target = ("assertion", "redundant", "table")
    elif donor_family == "opaque":
        target = ("opaque_token", "redundant", "prose")
    else:
        raise ValueError(f"unknown donor family: {donor_family}")
    matches = [
        row
        for row in rows
        if row["base_game_id"] == recipient["base_game_id"]
        and row["frame"] == recipient["frame"]
        and row["incentive"] == recipient["incentive"]
        and (row["surface_kind"], row["history"], row["mapping_format"]) == target
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {donor_family} donor, found {len(matches)}")
    return matches[0]
