from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
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
        raise ValueError("mean requires at least one value")
    return sum(map(float, values)) / len(values)


def _cluster_difference(
    rows: list[dict[str, Any]],
    left_filter: Any,
    right_filter: Any,
    left_key: str,
    right_key: str,
) -> tuple[list[Fraction], float]:
    clusters: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if left_filter(row):
            clusters[row["base_game_id"]][left_key].append(row["action_correct"])
        if right_filter(row):
            clusters[row["base_game_id"]][right_key].append(row["action_correct"])
    values = [
        Fraction(sum(cells[left_key]), len(cells[left_key]))
        - Fraction(sum(cells[right_key]), len(cells[right_key]))
        for cells in clusters.values()
    ]
    return values, exact_sign_flip_p(values)


def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload["rows"]
    for row in rows:
        row["action_correct"] = row["selected_action"] == row["expected_action"]
        row["rehearsed_correct"] = row["rehearsed_action"] == row["expected_action"]

    opaque = [row for row in rows if row["surface_kind"] == "opaque_token"]
    assertions = [row for row in rows if row["surface_kind"] == "assertion"]
    assertion_aligned = [row for row in assertions if row["incentive"] == "aligned"]
    assertion_opposed = [row for row in assertions if row["incentive"] == "opposed"]

    opaque_action = mean([row["action_correct"] for row in opaque])
    opaque_reports = mean(
        [report["correct"] for row in opaque for report in row["option_reports"].values()]
    )
    assertion_reports = mean(
        [
            report["correct"]
            for row in assertions
            for report in row["option_reports"].values()
        ]
    )
    aligned_accuracy = mean([row["action_correct"] for row in assertion_aligned])
    opposed_accuracy = mean([row["action_correct"] for row in assertion_opposed])
    replication_values, replication_p = _cluster_difference(
        assertions,
        lambda row: row["incentive"] == "aligned",
        lambda row: row["incentive"] == "opposed",
        "aligned",
        "opposed",
    )
    replication_gap = aligned_accuracy - opposed_accuracy
    replication_passed = (
        opaque_action >= 0.95
        and opaque_reports >= 0.95
        and assertion_reports >= 0.90
        and aligned_accuracy >= 0.90
        and opposed_accuracy <= 0.50
        and replication_gap >= 0.40
        and replication_p < 0.05
    )

    opposed = [row for row in rows if row["incentive"] == "opposed"]
    surface_accuracies = {
        surface: mean(
            [row["action_correct"] for row in opposed if row["surface_kind"] == surface]
        )
        for surface in ("assertion", "quoted_string", "labeled_button", "opaque_token")
    }
    nonassertive_lexical = (
        surface_accuracies["quoted_string"] + surface_accuracies["labeled_button"]
    ) / 2
    assertion_specific_values, assertion_specific_p = _cluster_difference(
        opposed,
        lambda row: row["surface_kind"] in {"quoted_string", "labeled_button"},
        lambda row: row["surface_kind"] == "assertion",
        "nonassertive",
        "assertion",
    )
    assertion_specific_gap = nonassertive_lexical - surface_accuracies["assertion"]
    assertion_specific = (
        surface_accuracies["assertion"] <= 0.50
        and nonassertive_lexical >= 0.80
        and assertion_specific_gap >= 0.25
        and assertion_specific_p < 0.05
    )
    opaque_surface_accuracy = surface_accuracies["opaque_token"]
    meaningful = ("assertion", "quoted_string", "labeled_button")
    lexical_capture = (
        opaque_surface_accuracy >= 0.90
        and all(surface_accuracies[surface] <= 0.65 for surface in meaningful)
        and all(
            opaque_surface_accuracy - surface_accuracies[surface] >= 0.20
            for surface in meaningful
        )
    )

    rehearsal_accuracy = mean([row["rehearsed_correct"] for row in assertion_opposed])
    rehearsal_gain = rehearsal_accuracy - opposed_accuracy
    rehearsal_clusters: dict[str, list[Fraction]] = defaultdict(list)
    for row in assertion_opposed:
        rehearsal_clusters[row["base_game_id"]].append(
            Fraction(int(row["rehearsed_correct"]) - int(row["action_correct"]))
        )
    rehearsal_values = [
        sum(values, Fraction(0)) / len(values) for values in rehearsal_clusters.values()
    ]
    rehearsal_p = exact_sign_flip_p(rehearsal_values)
    opaque_rehearsed = mean([row["rehearsed_correct"] for row in opaque])
    opaque_change = opaque_rehearsed - opaque_action
    rehearsal_promoted = (
        replication_passed
        and rehearsal_accuracy >= 0.75
        and rehearsal_gain >= 0.25
        and abs(opaque_change) <= 0.10
        and rehearsal_p < 0.05
    )

    correct_reports_wrong_direct = [
        row
        for row in assertion_opposed
        if all(report["correct"] for report in row["option_reports"].values())
        and not row["action_correct"]
    ]
    repaired = [row for row in correct_reports_wrong_direct if row["rehearsed_correct"]]
    content_equals_target = 0
    for row in correct_reports_wrong_direct:
        selected_index = next(
            int(index)
            for index, label in row["message_labels"].items()
            if label == row["selected_action"]
        )
        content_equals_target += row["concepts"][selected_index] == row["target_response"]

    frame_accuracy = {
        frame: mean(
            [row["action_correct"] for row in assertion_opposed if row["frame"] == frame]
        )
        for frame in ("strategic", "device")
    }
    formatting = mean(
        [
            item["formatting_compliant"]
            for row in rows
            for item in (row["action_result"], row["rehearsed_action_result"])
        ]
    )

    return {
        "schema_version": 1,
        "study_id": "V5-RBG-3",
        "source_run_id": payload["metadata"]["run_id"],
        "n": len(rows),
        "h1_replication": {
            "opaque_direct_action_accuracy": opaque_action,
            "opaque_option_report_accuracy": opaque_reports,
            "assertion_option_report_accuracy": assertion_reports,
            "assertion_aligned_action_accuracy": aligned_accuracy,
            "assertion_opposed_action_accuracy": opposed_accuracy,
            "aligned_minus_opposed_gap": replication_gap,
            "two_sided_exact_cluster_sign_flip_p": replication_p,
            "cluster_values": [float(value) for value in replication_values],
            "passed": replication_passed,
        },
        "h2_surface_localization": {
            "opposed_direct_action_accuracies": surface_accuracies,
            "nonassertive_lexical_minus_assertion_gap": assertion_specific_gap,
            "two_sided_exact_cluster_sign_flip_p": assertion_specific_p,
            "cluster_values": [float(value) for value in assertion_specific_values],
            "assertion_specific_signature": assertion_specific,
            "lexical_capture_signature": lexical_capture,
        },
        "h3_causal_rehearsal": {
            "opposed_assertion_direct_accuracy": opposed_accuracy,
            "opposed_assertion_rehearsed_accuracy": rehearsal_accuracy,
            "gain": rehearsal_gain,
            "two_sided_exact_paired_sign_flip_p": rehearsal_p,
            "opaque_direct_accuracy": opaque_action,
            "opaque_rehearsed_accuracy": opaque_rehearsed,
            "opaque_change": opaque_change,
            "promoted": rehearsal_promoted,
        },
        "diagnostics": {
            "both_reports_correct_direct_action_wrong": len(correct_reports_wrong_direct),
            "of_those_rehearsal_repaired": len(repaired),
            "of_those_selected_content_equals_target": content_equals_target,
            "opposed_assertion_direct_accuracy_by_frame": frame_accuracy,
            "forced_choice_formatting_compliance": formatting,
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
