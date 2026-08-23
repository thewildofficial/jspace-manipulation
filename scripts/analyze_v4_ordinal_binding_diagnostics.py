"""Post-hoc specificity checks for the prospective ordinal-binding result.

These diagnostics were designed after observing that both preregistered lure
tests moved by almost the same amount. They must never be described as
preregistered or confirmatory.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

RESPONSES = ("R1", "R2", "R3")


def exact_cluster_sign_flip(scores: list[float]) -> float:
    nonzero = [score for score in scores if abs(score) > 1e-12]
    if not nonzero:
        return 1.0
    if len(nonzero) > 24:
        raise ValueError("exact sign-flip test is capped at 24 nonzero clusters")
    observed = abs(math.fsum(nonzero))
    extreme = 0
    for mask in range(1 << len(nonzero)):
        permuted = math.fsum(
            score if mask & (1 << index) else -score for index, score in enumerate(nonzero)
        )
        extreme += abs(permuted) >= observed - 1e-12
    return extreme / (1 << len(nonzero))


def holm_two(left: float, right: float) -> tuple[float, float]:
    if left <= right:
        return min(1.0, 2 * left), max(min(1.0, 2 * left), right)
    adjusted_right, adjusted_left = holm_two(right, left)
    return adjusted_left, adjusted_right


def response_logit(row: dict[str, Any], response: str) -> float:
    label = next(label for label, surface in row["options"].items() if surface == response)
    return float(row["legal_action_logits"][label])


def chosen_response(row: dict[str, Any]) -> str:
    return str(row["options"][row["legal_choice"]])


def label_lure(row: dict[str, Any]) -> str:
    return f"R{'ABC'.index(str(row['selected_action'])) + 1}"


def position_lure(row: dict[str, Any]) -> str:
    certificate = row["ordinal_binding"]
    action = "ABC".index(str(row["selected_action"]))
    action_position = list(map(int, certificate["action_presentation_order"])).index(action)
    response = int(certificate["response_presentation_order"][action_position])
    return f"R{response + 1}"


def paired_rows(
    payload: dict[str, Any],
) -> tuple[dict[tuple[str, str], dict[str, Any]], list[str]]:
    rows = [row for row in payload["rows"] if row["decision_correct"]]
    indexed = {(str(row["condition_id"]), str(row["access_condition"])): row for row in rows}
    conditions = sorted({str(row["condition_id"]) for row in rows})
    if any(
        (condition, access) not in indexed
        for condition in conditions
        for access in ("retrospective", "answer_only")
    ):
        raise ValueError("incomplete within-decision access contrast")
    return indexed, conditions


def cluster_summary(cluster_values: dict[str, list[float]]) -> dict[str, Any]:
    scores = {
        cluster: math.fsum(values) / len(values)
        for cluster, values in sorted(cluster_values.items())
    }
    return {
        "clusters": len(scores),
        "cluster_mean": math.fsum(scores.values()) / len(scores),
        "two_sided_exact_cluster_sign_flip_p": exact_cluster_sign_flip(list(scores.values())),
        "positive_clusters": sum(score > 0 for score in scores.values()),
        "negative_clusters": sum(score < 0 for score in scores.values()),
        "zero_clusters": sum(abs(score) <= 1e-12 for score in scores.values()),
        "cluster_scores": scores,
    }


def specificity(
    indexed: dict[tuple[str, str], dict[str, Any]],
    conditions: list[str],
    lure_kind: str,
) -> dict[str, Any]:
    lure_function = label_lure if lure_kind == "label" else position_lure
    cluster_values: dict[str, list[float]] = defaultdict(list)
    eligible = 0
    for condition in conditions:
        retrospective = indexed[(condition, "retrospective")]
        answer_only = indexed[(condition, "answer_only")]
        true_response = str(retrospective["correct_surface"])
        lure = lure_function(retrospective)
        if lure == true_response:
            continue
        other_false = next(
            response for response in RESPONSES if response not in {true_response, lure}
        )
        score = (
            response_logit(retrospective, lure)
            - response_logit(answer_only, lure)
            - response_logit(retrospective, other_false)
            + response_logit(answer_only, other_false)
        )
        cluster_values[str(retrospective["base_game_id"])].append(score)
        eligible += 1
    return {
        "eligible_n": eligible,
        "contrast": (
            "rationale-minus-answer-only logit change for designated lure minus "
            "the same change for the nondesignated false response"
        ),
        **cluster_summary(cluster_values),
    }


def generic_false_vs_true(
    indexed: dict[tuple[str, str], dict[str, Any]], conditions: list[str]
) -> tuple[dict[str, Any], dict[str, dict[str, list[float]]]]:
    cluster_values: dict[str, list[float]] = defaultdict(list)
    by_frame: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for condition in conditions:
        retrospective = indexed[(condition, "retrospective")]
        answer_only = indexed[(condition, "answer_only")]
        true_response = str(retrospective["correct_surface"])
        false_scores = []
        for false_response in RESPONSES:
            if false_response == true_response:
                continue
            false_scores.append(
                response_logit(retrospective, false_response)
                - response_logit(retrospective, true_response)
                - response_logit(answer_only, false_response)
                + response_logit(answer_only, true_response)
            )
        score = math.fsum(false_scores) / len(false_scores)
        cluster = str(retrospective["base_game_id"])
        frame = str(retrospective["frame"])
        cluster_values[cluster].append(score)
        by_frame[cluster][frame].append(score)
    return cluster_summary(cluster_values), by_frame


def frame_interaction(
    indexed: dict[tuple[str, str], dict[str, Any]],
    conditions: list[str],
    generic_by_frame: dict[str, dict[str, list[float]]],
) -> dict[str, Any]:
    logit_clusters: dict[str, list[float]] = {}
    for cluster, frames in generic_by_frame.items():
        strategic = math.fsum(frames["strategic"]) / len(frames["strategic"])
        nonagentic = math.fsum(frames["nonagentic"]) / len(frames["nonagentic"])
        logit_clusters[cluster] = [strategic - nonagentic]

    accuracy: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    answer_only_only_by_frame: Counter[str] = Counter()
    for condition in conditions:
        retrospective = indexed[(condition, "retrospective")]
        answer_only = indexed[(condition, "answer_only")]
        frame = str(retrospective["frame"])
        cluster = str(retrospective["base_game_id"])
        score = float(bool(retrospective["legal_choice_correct"])) - float(
            bool(answer_only["legal_choice_correct"])
        )
        accuracy[cluster][frame].append(score)
        if answer_only["legal_choice_correct"] and not retrospective["legal_choice_correct"]:
            answer_only_only_by_frame[frame] += 1
    accuracy_clusters = {
        cluster: [
            math.fsum(frames["strategic"]) / len(frames["strategic"])
            - math.fsum(frames["nonagentic"]) / len(frames["nonagentic"])
        ]
        for cluster, frames in accuracy.items()
    }
    return {
        "direction": "strategic minus matched nonagentic rationale effect",
        "generic_false_vs_true_logit_interaction": cluster_summary(logit_clusters),
        "accuracy_penalty_interaction": cluster_summary(accuracy_clusters),
        "answer_only_only_errors_by_frame": dict(answer_only_only_by_frame),
    }


def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    indexed, conditions = paired_rows(payload)
    label = specificity(indexed, conditions, "label")
    position = specificity(indexed, conditions, "position")
    label_adjusted, position_adjusted = holm_two(
        float(label["two_sided_exact_cluster_sign_flip_p"]),
        float(position["two_sided_exact_cluster_sign_flip_p"]),
    )
    label["exploratory_holm_p_across_two_specificity_checks"] = label_adjusted
    position["exploratory_holm_p_across_two_specificity_checks"] = position_adjusted
    generic, generic_by_frame = generic_false_vs_true(indexed, conditions)

    overlap: Counter[str] = Counter()
    for condition in conditions:
        row = indexed[(condition, "retrospective")]
        label_value = label_lure(row)
        position_value = position_lure(row)
        true_value = str(row["correct_surface"])
        key = (
            f"label_equals_position={label_value == position_value};"
            f"label_equals_true={label_value == true_value};"
            f"position_equals_true={position_value == true_value}"
        )
        overlap[key] += 1

    return {
        "status": "posthoc_exploratory_after_inspecting_preregistered_result",
        "reason": (
            "Both preregistered lure-vs-true tests had nearly identical effects. "
            "Because either test can become positive when the rationale generically "
            "suppresses the true response relative to both false responses, these "
            "checks compare each designated lure with the other false response."
        ),
        "source_metadata": payload["metadata"],
        "eligible_correct_decisions": len(conditions),
        "lure_overlap_counts": dict(overlap),
        "generic_false_vs_true_pull": generic,
        "designated_lure_specificity": {
            "action_label": label,
            "action_position": position,
        },
        "strategic_frame_diagnostic": frame_interaction(indexed, conditions, generic_by_frame),
        "claim_boundary": (
            "These analyses were specified after result inspection. They can reject "
            "or weaken a mechanistic interpretation, but cannot confirm a new one."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rendered = json.dumps(analyze(payload), indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            raise RuntimeError(f"refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
