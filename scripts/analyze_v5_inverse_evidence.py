from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Any


def exact_sign_flip_p(values: list[Fraction]) -> float:
    nonzero = [value for value in values if value]
    if not nonzero:
        return 1.0
    observed = abs(sum(nonzero))
    counts: Counter[Fraction] = Counter({Fraction(0): 1})
    for value in nonzero:
        updated: Counter[Fraction] = Counter()
        for total, count in counts.items():
            updated[total + value] += count
            updated[total - value] += count
        counts = updated
    extreme = sum(count for total, count in counts.items() if abs(total) >= observed)
    return extreme / (2 ** len(nonzero))


def mean(values: list[bool | float]) -> float:
    if not values:
        raise ValueError("mean requires values")
    return sum(map(float, values)) / len(values)


def report_accuracy(rows: list[dict[str, Any]]) -> float:
    return mean(
        [report["correct"] for row in rows for report in row["option_reports"].values()]
    )


def cluster_contrast(
    rows: list[dict[str, Any]],
    filters_and_signs: list[tuple[Callable[[dict[str, Any]], bool], int]],
) -> tuple[list[Fraction], float]:
    clusters: dict[str, list[list[bool]]] = defaultdict(
        lambda: [[] for _ in filters_and_signs]
    )
    for row in rows:
        for index, (predicate, _) in enumerate(filters_and_signs):
            if predicate(row):
                clusters[row["base_game_id"]][index].append(row["action_correct"])
    values = []
    for cells in clusters.values():
        value = Fraction(0)
        for cell, (_, sign) in zip(cells, filters_and_signs, strict=True):
            value += sign * Fraction(sum(cell), len(cell))
        values.append(value)
    return values, exact_sign_flip_p(values)


def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload["rows"]
    opaque = [row for row in rows if row["surface_kind"] == "opaque_token"]
    all_reports = report_accuracy(rows)
    opaque_actions = mean([row["action_correct"] for row in opaque])
    h1_passed = opaque_actions >= 0.95 and all_reports >= 0.95

    assertion_opposed = [
        row
        for row in rows
        if row["surface_kind"] == "assertion" and row["incentive"] == "opposed"
    ]
    no_history = [row for row in assertion_opposed if row["history"] == "none"]
    redundant = [row for row in assertion_opposed if row["history"] == "redundant"]
    no_history_accuracy = mean([row["action_correct"] for row in no_history])
    redundant_accuracy = mean([row["action_correct"] for row in redundant])
    history_gap = no_history_accuracy - redundant_accuracy
    report_gap = report_accuracy(no_history) - report_accuracy(redundant)
    history_values, history_p = cluster_contrast(
        assertion_opposed,
        [
            (lambda row: row["history"] == "none", 1),
            (lambda row: row["history"] == "redundant", -1),
        ],
    )
    h2_promoted = (
        h1_passed
        and history_gap >= 0.20
        and abs(report_gap) <= 0.05
        and history_p < 0.05
    )

    opposed = [row for row in rows if row["incentive"] == "opposed"]
    did_values, did_p = cluster_contrast(
        opposed,
        [
            (
                lambda row: row["surface_kind"] == "assertion"
                and row["history"] == "none",
                1,
            ),
            (
                lambda row: row["surface_kind"] == "assertion"
                and row["history"] == "redundant",
                -1,
            ),
            (
                lambda row: row["surface_kind"] == "opaque_token"
                and row["history"] == "none",
                -1,
            ),
            (
                lambda row: row["surface_kind"] == "opaque_token"
                and row["history"] == "redundant",
                1,
            ),
        ],
    )
    difference_in_differences = float(sum(did_values) / len(did_values))
    h3_promoted = (
        h1_passed and difference_in_differences >= 0.20 and did_p < 0.05
    )

    target_cell = [
        row
        for row in rows
        if row["frame"] == "strategic"
        and row["incentive"] == "opposed"
        and row["surface_kind"] == "assertion"
        and row["history"] == "redundant"
        and row["mapping_format"] == "prose"
    ]
    target_opaque = [
        row
        for row in rows
        if row["frame"] == "strategic"
        and row["incentive"] == "opposed"
        and row["surface_kind"] == "opaque_token"
        and row["history"] == "redundant"
        and row["mapping_format"] == "prose"
    ]
    target_accuracy = mean([row["action_correct"] for row in target_cell])
    target_reports = report_accuracy(target_cell)
    target_opaque_accuracy = mean([row["action_correct"] for row in target_opaque])
    h4_passed = (
        target_accuracy <= 0.50
        and target_reports >= 0.90
        and target_opaque_accuracy >= 0.90
    )

    history_by_mapping = {}
    for mapping in ("prose", "table"):
        for history in ("none", "redundant"):
            cell = [
                row
                for row in assertion_opposed
                if row["mapping_format"] == mapping and row["history"] == history
            ]
            history_by_mapping[f"{mapping}:{history}"] = mean(
                [row["action_correct"] for row in cell]
            )
    aligned_assertions = [
        row
        for row in rows
        if row["surface_kind"] == "assertion" and row["incentive"] == "aligned"
    ]
    reportable_errors = [
        row
        for row in assertion_opposed
        if not row["action_correct"]
        and all(report["correct"] for report in row["option_reports"].values())
    ]
    content_equals_target = 0
    for row in reportable_errors:
        selected_index = next(
            int(index)
            for index, label in row["message_labels"].items()
            if label == row["selected_action"]
        )
        content_equals_target += row["concepts"][selected_index] == row["target_response"]

    return {
        "schema_version": 1,
        "study_id": "V5-RBG-4",
        "source_run_id": payload["metadata"]["run_id"],
        "n": len(rows),
        "h1_control_competence": {
            "opaque_action_accuracy": opaque_actions,
            "all_option_report_accuracy": all_reports,
            "passed": h1_passed,
        },
        "h2_inverse_evidence": {
            "no_history_assertion_opposed_accuracy": no_history_accuracy,
            "redundant_history_assertion_opposed_accuracy": redundant_accuracy,
            "action_gap": history_gap,
            "option_report_accuracy_gap": report_gap,
            "two_sided_exact_cluster_sign_flip_p": history_p,
            "cluster_values": [float(value) for value in history_values],
            "promoted": h2_promoted,
        },
        "h3_semantic_specificity": {
            "history_harm_assertion_minus_opaque": difference_in_differences,
            "two_sided_exact_cluster_sign_flip_p": did_p,
            "cluster_values": [float(value) for value in did_values],
            "promoted": h3_promoted,
        },
        "h4_rbg2_cell_replication": {
            "strategic_opposed_redundant_prose_assertion_accuracy": target_accuracy,
            "option_report_accuracy": target_reports,
            "matched_opaque_action_accuracy": target_opaque_accuracy,
            "passed": h4_passed,
        },
        "diagnostics": {
            "opposed_assertion_accuracy_by_mapping_and_history": history_by_mapping,
            "aligned_assertion_action_accuracy": mean(
                [row["action_correct"] for row in aligned_assertions]
            ),
            "both_reports_correct_but_action_wrong": len(reportable_errors),
            "of_those_selected_content_equals_target": content_equals_target,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = analyze(json.loads(args.input.read_text(encoding="utf-8")))
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite analysis: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
