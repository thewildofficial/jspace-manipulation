from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

CONTROLS = (
    "answer_only",
    "matched_trajectory",
    "ordinal_trajectory",
    "reconstruction",
)


def exact_mcnemar(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    lower = min(left_only, right_only)
    tail = sum(math.comb(discordant, value) for value in range(lower + 1))
    return min(1.0, 2 * tail / (2**discordant))


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * p_value))
        adjusted[name] = running
    return adjusted


def accuracy(rows: list[dict[str, Any]]) -> float:
    return sum(bool(row["legal_choice_correct"]) for row in rows) / len(rows)


def correct_logit_margin(row: dict[str, Any]) -> float:
    logits = {str(key): float(value) for key, value in row["legal_action_logits"].items()}
    expected = str(row["expected_label"])
    return logits[expected] - max(
        value for label, value in logits.items() if label != expected
    )


def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in payload["rows"] if row["decision_correct"]]
    indexed = {
        (row["condition_id"], row["query_type"], row["access_condition"]): row
        for row in rows
    }
    queries = sorted({row["query_type"] for row in rows})
    accesses = ("retrospective", *CONTROLS)
    cells: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    raw_p: dict[str, float] = {}

    for query in queries:
        query_rows = [row for row in rows if row["query_type"] == query]
        condition_ids = sorted({row["condition_id"] for row in query_rows})
        for access in accesses:
            selected = [
                row for row in query_rows if row["access_condition"] == access
            ]
            cells[f"{query}:{access}"] = {
                "n": len(selected),
                "legal_choice_accuracy": accuracy(selected),
                "formatting_compliance": sum(
                    row["formatting_compliant"] for row in selected
                )
                / len(selected),
                "choice_distribution": dict(
                    sorted(Counter(row["legal_choice"] for row in selected).items())
                ),
                "mean_correct_logit_margin": sum(
                    correct_logit_margin(row) for row in selected
                )
                / len(selected),
            }

        for control in CONTROLS:
            retro_only = control_only = both_correct = both_wrong = 0
            margin_differences = []
            for condition_id in condition_ids:
                retro = indexed[(condition_id, query, "retrospective")]
                other = indexed[(condition_id, query, control)]
                left = bool(retro["legal_choice_correct"])
                right = bool(other["legal_choice_correct"])
                if left and right:
                    both_correct += 1
                elif left:
                    retro_only += 1
                elif right:
                    control_only += 1
                else:
                    both_wrong += 1
                margin_differences.append(
                    correct_logit_margin(retro) - correct_logit_margin(other)
                )
            key = f"{query}:retrospective_vs_{control}"
            p_value = exact_mcnemar(retro_only, control_only)
            raw_p[key] = p_value
            paired[key] = {
                "n": len(condition_ids),
                "retrospective_only_correct": retro_only,
                "control_only_correct": control_only,
                "both_correct": both_correct,
                "both_wrong": both_wrong,
                "accuracy_difference": (
                    cells[f"{query}:retrospective"]["legal_choice_accuracy"]
                    - cells[f"{query}:{control}"]["legal_choice_accuracy"]
                ),
                "mean_correct_logit_margin_difference": (
                    sum(margin_differences) / len(margin_differences)
                ),
                "exact_mcnemar_p": p_value,
            }

    adjusted = holm_adjust(raw_p)
    for name, value in adjusted.items():
        paired[name]["holm_p_across_12_tests"] = value
    primary_p = {
        name: value
        for name, value in raw_p.items()
        if name.startswith("predicted_response:")
    }
    for name, value in holm_adjust(primary_p).items():
        paired[name]["holm_p_primary_four_tests"] = value

    frame_cells = {
        f"{query}:{frame}:{access}": {
            "n": len(selected := [
                row
                for row in rows
                if row["query_type"] == query
                and row["frame"] == frame
                and row["access_condition"] == access
            ]),
            "legal_choice_accuracy": accuracy(selected),
        }
        for query in queries
        for frame in sorted({row["frame"] for row in rows})
        for access in accesses
    }
    return {
        "metadata": payload["metadata"],
        "eligible_decisions": len({row["condition_id"] for row in rows}),
        "report_rows": len(rows),
        "cells": cells,
        "paired_legal_choice_tests": paired,
        "frame_cells": frame_cells,
        "decision_accuracy_was_conditioned_on": True,
        "inference_note": (
            "Primary binary inference is paired exact McNemar for predicted_response "
            "with Holm correction over four controls. The 12-test Holm family is also "
            "reported; other queries and logit margins are exploratory/descriptive."
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
