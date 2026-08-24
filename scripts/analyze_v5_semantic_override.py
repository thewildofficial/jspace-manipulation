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


def mean(values: list[bool | float]) -> float | None:
    return sum(map(float, values)) / len(values) if values else None


def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload["rows"]
    opaque_control = [
        row
        for row in rows
        if row["policy_access"] == "explicit"
        and row["message_semantics"] == "opaque_tokens"
    ]
    control_cells = {}
    for frame in ("strategic", "device"):
        for incentive in ("aligned", "opposed"):
            for receiver_type in ("literal", "contrarian"):
                cell = [
                    row
                    for row in opaque_control
                    if row["frame"] == frame
                    and row["incentive"] == incentive
                    and row["receiver_type"] == receiver_type
                ]
                control_cells[f"{frame}:{incentive}:{receiver_type}"] = mean(
                    [row["action_correct"] for row in cell]
                )
    opaque_action_accuracy = float(mean([row["action_correct"] for row in opaque_control]))
    opaque_report_accuracy = float(
        mean(
            [
                report["correct"]
                for row in opaque_control
                for report in row["option_reports"].values()
            ]
        )
    )
    minimum_control_cell = min(control_cells.values())
    h1_passed = (
        opaque_action_accuracy >= 0.90
        and minimum_control_cell >= 0.80
        and opaque_report_accuracy >= 0.90
    )

    explicit_contrarian = [
        row
        for row in rows
        if row["policy_access"] == "explicit" and row["receiver_type"] == "contrarian"
    ]
    claim_rows = [
        row for row in explicit_contrarian if row["message_semantics"] == "claims"
    ]
    token_rows = [
        row
        for row in explicit_contrarian
        if row["message_semantics"] == "opaque_tokens"
    ]
    claim_accuracy = float(mean([row["action_correct"] for row in claim_rows]))
    token_accuracy = float(mean([row["action_correct"] for row in token_rows]))
    claim_report_accuracy = float(
        mean(
            [
                report["correct"]
                for row in claim_rows
                for report in row["option_reports"].values()
            ]
        )
    )
    clusters: dict[str, dict[str, list[bool]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in explicit_contrarian:
        clusters[row["base_game_id"]][row["message_semantics"]].append(
            row["action_correct"]
        )
    cluster_values = [
        Fraction(
            sum(cells["opaque_tokens"]), len(cells["opaque_tokens"])
        )
        - Fraction(sum(cells["claims"]), len(cells["claims"]))
        for cells in clusters.values()
    ]
    gap = token_accuracy - claim_accuracy
    p_value = exact_sign_flip_p(cluster_values)
    h2_promoted = (
        h1_passed
        and token_accuracy >= 0.85
        and claim_accuracy <= 0.65
        and claim_report_accuracy >= 0.85
        and gap >= 0.20
        and p_value < 0.05
    )
    both_reports_correct_action_wrong = [
        row
        for row in claim_rows
        if all(report["correct"] for report in row["option_reports"].values())
        and not row["action_correct"]
    ]
    semantic_target_errors = 0
    for row in both_reports_correct_action_wrong:
        selected_index = next(
            int(index)
            for index, label in row["message_labels"].items()
            if label == row["selected_action"]
        )
        semantic_target_errors += row["concepts"][selected_index] == row["target_response"]

    cells = {}
    factors = (
        ("frame", ("strategic", "device")),
        ("incentive", ("aligned", "opposed")),
        ("receiver_type", ("literal", "contrarian")),
        ("policy_access", ("inferred", "explicit")),
        ("message_semantics", ("claims", "opaque_tokens")),
    )
    for row in rows:
        key = ":".join(str(row[name]) for name, _ in factors)
        cells.setdefault(key, {"n": 0, "actions_correct": 0, "reports_correct": 0})
        cells[key]["n"] += 1
        cells[key]["actions_correct"] += row["action_correct"]
        cells[key]["reports_correct"] += sum(
            report["correct"] for report in row["option_reports"].values()
        )
    for cell in cells.values():
        cell["action_accuracy"] = cell["actions_correct"] / cell["n"]
        cell["option_report_accuracy"] = cell["reports_correct"] / (2 * cell["n"])

    return {
        "schema_version": 1,
        "study_id": "V5-RBG-2",
        "source_run_id": payload["metadata"]["run_id"],
        "n": len(rows),
        "h1_explicit_opaque_competence": {
            "action_accuracy": opaque_action_accuracy,
            "minimum_cell_accuracy": minimum_control_cell,
            "option_report_accuracy": opaque_report_accuracy,
            "passed": h1_passed,
            "cells": control_cells,
        },
        "h2_semantic_action_override": {
            "explicit_contrarian_opaque_action_accuracy": token_accuracy,
            "explicit_contrarian_claim_action_accuracy": claim_accuracy,
            "opaque_minus_claim_gap": gap,
            "claim_option_report_accuracy": claim_report_accuracy,
            "two_sided_exact_cluster_sign_flip_p": p_value,
            "cluster_values": [float(value) for value in cluster_values],
            "promoted": h2_promoted,
        },
        "diagnostics": {
            "explicit_contrarian_claim_rows": len(claim_rows),
            "both_reports_correct_but_action_wrong": len(
                both_reports_correct_action_wrong
            ),
            "of_those_claim_content_equals_target": semantic_target_errors,
        },
        "cells": cells,
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
