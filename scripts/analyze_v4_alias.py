from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def exact_mcnemar(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    lower = min(left_only, right_only)
    tail = sum(math.comb(discordant, value) for value in range(lower + 1))
    return min(1.0, 2 * tail / (2**discordant))


def margin(row: dict[str, Any]) -> float:
    expected = str(row["expected_label"])
    logits = {str(key): float(value) for key, value in row["legal_action_logits"].items()}
    return logits[expected] - max(value for key, value in logits.items() if key != expected)


def chosen_response(row: dict[str, Any]) -> str:
    surface = str(row["options"][row["legal_choice"]])
    if row["response_naming"] == "indexed":
        return surface
    aliases = list(map(str, row["response_aliases"]))
    return f"R{aliases.index(surface) + 1}"


def paired(
    indexed: dict[tuple[str, str, str], dict[str, Any]],
    conditions: list[str],
    left: tuple[str, str],
    right: tuple[str, str],
) -> dict[str, Any]:
    left_only = right_only = both = neither = 0
    margin_differences = []
    for condition in conditions:
        left_row = indexed[(condition, *left)]
        right_row = indexed[(condition, *right)]
        left_correct = bool(left_row["legal_choice_correct"])
        right_correct = bool(right_row["legal_choice_correct"])
        if left_correct and right_correct:
            both += 1
        elif left_correct:
            left_only += 1
        elif right_correct:
            right_only += 1
        else:
            neither += 1
        margin_differences.append(margin(left_row) - margin(right_row))
    return {
        "n": len(conditions),
        "left": {"access": left[0], "response_naming": left[1]},
        "right": {"access": right[0], "response_naming": right[1]},
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "both_correct": both,
        "both_wrong": neither,
        "accuracy_difference": (left_only - right_only) / len(conditions),
        "mean_correct_logit_margin_difference": (
            sum(margin_differences) / len(margin_differences)
        ),
        "two_sided_exact_mcnemar_p": exact_mcnemar(left_only, right_only),
    }


def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in payload["rows"] if row["decision_correct"]]
    indexed = {
        (row["condition_id"], row["access_condition"], row["response_naming"]): row
        for row in rows
    }
    conditions = sorted({row["condition_id"] for row in rows})
    cells = {}
    for access in ("retrospective", "answer_only"):
        for naming in ("indexed", "arbitrary_alias"):
            selected = [
                indexed[(condition, access, naming)] for condition in conditions
            ]
            errors = [row for row in selected if not row["legal_choice_correct"]]
            cells[f"{access}:{naming}"] = {
                "n": len(selected),
                "legal_choice_accuracy": sum(
                    row["legal_choice_correct"] for row in selected
                )
                / len(selected),
                "mean_correct_logit_margin": sum(map(margin, selected)) / len(selected),
                "wrong_choice_response_distribution": dict(
                    sorted(Counter(map(chosen_response, errors)).items())
                ),
            }

    comparisons = {
        "primary_alias_rescue_with_rationale": paired(
            indexed,
            conditions,
            ("retrospective", "arbitrary_alias"),
            ("retrospective", "indexed"),
        ),
        "alias_effect_answer_only": paired(
            indexed,
            conditions,
            ("answer_only", "arbitrary_alias"),
            ("answer_only", "indexed"),
        ),
        "trajectory_penalty_indexed": paired(
            indexed,
            conditions,
            ("retrospective", "indexed"),
            ("answer_only", "indexed"),
        ),
        "trajectory_penalty_alias": paired(
            indexed,
            conditions,
            ("retrospective", "arbitrary_alias"),
            ("answer_only", "arbitrary_alias"),
        ),
    }
    retro_alias = comparisons["primary_alias_rescue_with_rationale"]
    answer_alias = comparisons["alias_effect_answer_only"]
    return {
        "metadata": payload["metadata"],
        "eligible_decisions": len(conditions),
        "report_rows": len(rows),
        "cells": cells,
        "paired_tests": comparisons,
        "accuracy_difference_in_differences": (
            retro_alias["accuracy_difference"]
            - answer_alias["accuracy_difference"]
        ),
        "logit_margin_difference_in_differences": (
            retro_alias["mean_correct_logit_margin_difference"]
            - answer_alias["mean_correct_logit_margin_difference"]
        ),
        "interpretation": (
            "The ordinal-binding account requires alias rescue in retrospective "
            "reports without a comparable answer-only improvement."
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
