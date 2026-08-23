from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

CONTROLS = ("answer_only", "matched_trajectory", "reconstruction")


def exact_mcnemar(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value) for value in range(min(left_only, right_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2 * tail)


def accuracy(rows: list[dict[str, Any]], field: str = "legal_choice_correct") -> float:
    return sum(bool(row[field]) for row in rows) / len(rows)


def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    decisions = payload["rows"]
    reports = [row for row in payload["self_report_rows"] if row["decision_correct"]]
    by_report = {
        (row["condition_id"], row["query_type"], row["access_condition"]): row
        for row in reports
    }
    queries = sorted({row["query_type"] for row in reports})

    report_cells: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    frame_cells: dict[str, Any] = {}
    copyability: dict[str, Any] = {}
    label_diagnostics: dict[str, Any] = {}
    for query in queries:
        query_rows = [row for row in reports if row["query_type"] == query]
        for access in ("retrospective", *CONTROLS):
            selected = [row for row in query_rows if row["access_condition"] == access]
            report_cells[f"{query}:{access}"] = {
                "n": len(selected),
                "legal_choice_accuracy": accuracy(selected),
                "formatting_compliance": sum(
                    row["formatting_compliant"] for row in selected
                )
                / len(selected),
                "choice_distribution": dict(Counter(row["legal_choice"] for row in selected)),
                "repeats_decision_label": sum(
                    row["legal_choice"] == row["selected_action"] for row in selected
                )
                / len(selected),
            }

        condition_ids = sorted(
            {
                row["condition_id"]
                for row in query_rows
                if row["access_condition"] == "retrospective"
            }
        )
        for control in CONTROLS:
            retro_only = 0
            control_only = 0
            both_correct = 0
            both_wrong = 0
            for condition_id in condition_ids:
                retro = by_report[(condition_id, query, "retrospective")][
                    "legal_choice_correct"
                ]
                other = by_report[(condition_id, query, control)]["legal_choice_correct"]
                if retro and other:
                    both_correct += 1
                elif retro:
                    retro_only += 1
                elif other:
                    control_only += 1
                else:
                    both_wrong += 1
            paired[f"{query}:retrospective_vs_{control}"] = {
                "n": len(condition_ids),
                "retrospective_only_correct": retro_only,
                "control_only_correct": control_only,
                "both_correct": both_correct,
                "both_wrong": both_wrong,
                "accuracy_difference": (
                    report_cells[f"{query}:retrospective"]["legal_choice_accuracy"]
                    - report_cells[f"{query}:{control}"]["legal_choice_accuracy"]
                ),
                "exact_mcnemar_p": exact_mcnemar(retro_only, control_only),
            }

        for frame in sorted({row["frame"] for row in query_rows}):
            for access in ("retrospective", *CONTROLS):
                selected = [
                    row
                    for row in query_rows
                    if row["frame"] == frame and row["access_condition"] == access
                ]
                frame_cells[f"{query}:{frame}:{access}"] = {
                    "n": len(selected),
                    "legal_choice_accuracy": accuracy(selected),
                }

        retrospective = [
            row for row in query_rows if row["access_condition"] == "retrospective"
        ]
        copyability[query] = {
            str(present).lower(): {
                "n": len(selected := [
                    row
                    for row in retrospective
                    if row["target_surface_in_trajectory"] is present
                ]),
                "legal_choice_accuracy": accuracy(selected) if selected else None,
            }
            for present in (False, True)
        }
        label_diagnostics[query] = {
            "expected_distribution": dict(
                Counter(row["expected_label"] for row in retrospective)
            ),
            "decision_label_distribution": dict(
                Counter(row["selected_action"] for row in retrospective)
            ),
        }

    behavior_errors = [
        {
            "condition_id": row["condition_id"],
            "split": row["split"],
            "frame": row["frame"],
            "pair_type": row["pair_type"],
            "expected_action": row["expected_action"],
            "selected_action": row["selected_action"],
            "generated_tokens": row["generated_tokens"],
            "generated_text": row["generated_text"],
        }
        for row in decisions
        if not row["correct"]
    ]
    by_split = defaultdict(list)
    for row in decisions:
        by_split[row["split"]].append(row)

    return {
        "behavior": {
            "n": len(decisions),
            "parseability": sum(row["parseable"] for row in decisions) / len(decisions),
            "accuracy": accuracy(decisions, "correct"),
            "token_ceiling_hits": sum(row["hit_token_ceiling"] for row in decisions),
            "split_accuracy": {
                split: accuracy(rows, "correct") for split, rows in sorted(by_split.items())
            },
            "errors": behavior_errors,
        },
        "reports": {
            "eligible_decisions": len(
                {row["condition_id"] for row in reports}
            ),
            "cells": report_cells,
            "paired_legal_choice_tests": paired,
            "frame_cells": frame_cells,
            "copyability": copyability,
            "label_diagnostics": label_diagnostics,
        },
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
