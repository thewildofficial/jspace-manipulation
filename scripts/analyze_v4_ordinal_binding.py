from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any


def exact_mcnemar(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    lower = min(left_only, right_only)
    tail = sum(math.comb(discordant, value) for value in range(lower + 1))
    return min(1.0, 2 * tail / (2**discordant))


def exact_cluster_sign_flip(cluster_scores: list[float]) -> float:
    """Two-sided randomization p-value under cluster-level sign exchangeability."""
    scores = [score for score in cluster_scores if abs(score) > 1e-12]
    if not scores:
        return 1.0
    if len(scores) > 24:
        raise ValueError("exact sign-flip test is intentionally capped at 24 clusters")
    observed = abs(math.fsum(scores))
    extreme = 0
    total = 1 << len(scores)
    for mask in range(total):
        permuted = math.fsum(
            score if mask & (1 << index) else -score
            for index, score in enumerate(scores)
        )
        if abs(permuted) >= observed - 1e-12:
            extreme += 1
    return extreme / total


def holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=lambda key: (p_values[key], key))
    adjusted: dict[str, float] = {}
    running = 0.0
    size = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (size - rank) * p_values[key]))
        adjusted[key] = running
    return adjusted


def chosen_response(row: dict[str, Any]) -> str:
    return str(row["options"][row["legal_choice"]])


def relative_logit(row: dict[str, Any], response: str) -> float:
    label_for_response = next(
        label for label, surface in row["options"].items() if surface == response
    )
    expected = str(row["expected_label"])
    logits = {str(key): float(value) for key, value in row["legal_action_logits"].items()}
    return logits[label_for_response] - logits[expected]


def label_lure(row: dict[str, Any]) -> str:
    return f"R{'ABC'.index(str(row['selected_action'])) + 1}"


def position_lure(row: dict[str, Any]) -> str:
    certificate = row["ordinal_binding"]
    action = "ABC".index(str(row["selected_action"]))
    action_position = list(map(int, certificate["action_presentation_order"])).index(action)
    response = int(certificate["response_presentation_order"][action_position])
    return f"R{response + 1}"


def paired_accuracy(
    indexed: dict[tuple[str, str], dict[str, Any]], conditions: list[str]
) -> dict[str, Any]:
    retrospective_only = answer_only_only = both = neither = 0
    for condition in conditions:
        retrospective = bool(indexed[(condition, "retrospective")]["legal_choice_correct"])
        answer_only = bool(indexed[(condition, "answer_only")]["legal_choice_correct"])
        if retrospective and answer_only:
            both += 1
        elif retrospective:
            retrospective_only += 1
        elif answer_only:
            answer_only_only += 1
        else:
            neither += 1
    return {
        "n": len(conditions),
        "retrospective_only_correct": retrospective_only,
        "answer_only_only_correct": answer_only_only,
        "both_correct": both,
        "both_wrong": neither,
        "accuracy_difference": (retrospective_only - answer_only_only) / len(conditions),
        "two_sided_exact_mcnemar_p_unclustered_secondary": exact_mcnemar(
            retrospective_only, answer_only_only
        ),
    }


def lure_test(
    indexed: dict[tuple[str, str], dict[str, Any]],
    conditions: list[str],
    lure: Callable[[dict[str, Any]], str],
) -> dict[str, Any]:
    cluster_values: dict[str, list[float]] = defaultdict(list)
    retrospective_only_lure = answer_only_only_lure = both_lure = neither_lure = 0
    eligible = []
    for condition in conditions:
        retrospective = indexed[(condition, "retrospective")]
        answer_only = indexed[(condition, "answer_only")]
        lure_response = lure(retrospective)
        if lure_response == retrospective["correct_surface"]:
            continue
        eligible.append(condition)
        delta = relative_logit(retrospective, lure_response) - relative_logit(
            answer_only, lure_response
        )
        cluster_values[str(retrospective["base_game_id"])].append(delta)
        retro_chose = chosen_response(retrospective) == lure_response
        answer_chose = chosen_response(answer_only) == lure_response
        if retro_chose and answer_chose:
            both_lure += 1
        elif retro_chose:
            retrospective_only_lure += 1
        elif answer_chose:
            answer_only_only_lure += 1
        else:
            neither_lure += 1
    cluster_scores = {
        cluster: math.fsum(values) / len(values)
        for cluster, values in sorted(cluster_values.items())
    }
    return {
        "eligible_n": len(eligible),
        "clusters": len(cluster_scores),
        "mean_rationale_induced_lure_logit_pull": (
            math.fsum(cluster_scores.values()) / len(cluster_scores)
        ),
        "two_sided_exact_cluster_sign_flip_p": exact_cluster_sign_flip(
            list(cluster_scores.values())
        ),
        "cluster_scores": cluster_scores,
        "forced_choice_lure_intrusions": {
            "retrospective_only": retrospective_only_lure,
            "answer_only_only": answer_only_only_lure,
            "both": both_lure,
            "neither": neither_lure,
            "two_sided_exact_mcnemar_p_unclustered_secondary": exact_mcnemar(
                retrospective_only_lure, answer_only_only_lure
            ),
        },
    }


def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in payload["rows"] if row["decision_correct"]]
    indexed = {
        (str(row["condition_id"]), str(row["access_condition"])): row for row in rows
    }
    conditions = sorted({str(row["condition_id"]) for row in rows})
    if any(
        (condition, access) not in indexed
        for condition in conditions
        for access in ("retrospective", "answer_only")
    ):
        raise ValueError("incomplete within-decision access contrast")
    tests = {
        "action_label_to_response_label_lure": lure_test(
            indexed, conditions, label_lure
        ),
        "action_position_to_response_position_lure": lure_test(
            indexed, conditions, position_lure
        ),
    }
    raw_p = {
        name: float(result["two_sided_exact_cluster_sign_flip_p"])
        for name, result in tests.items()
    }
    adjusted = holm(raw_p)
    for name, value in adjusted.items():
        tests[name]["holm_p_across_two_primary_lures"] = value
    frame_accuracy = {
        frame: paired_accuracy(
            indexed,
            [
                condition
                for condition in conditions
                if indexed[(condition, "retrospective")]["frame"] == frame
            ],
        )
        for frame in ("strategic", "nonagentic")
    }
    return {
        "metadata": payload["metadata"],
        "eligible_correct_decisions": len(conditions),
        "base_game_clusters": len(
            {
                str(indexed[(condition, "retrospective")]["base_game_id"])
                for condition in conditions
            }
        ),
        "primary_tests": tests,
        "trajectory_accuracy_penalty": paired_accuracy(indexed, conditions),
        "trajectory_accuracy_penalty_by_frame": frame_accuracy,
        "interpretation": (
            "Label-only support indicates cross-role lexical binding; position-only "
            "support indicates an ordering-ID collision; support for both indicates "
            "a mixed binding code; neither with a surviving accuracy penalty indicates "
            "generic relational interference."
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
