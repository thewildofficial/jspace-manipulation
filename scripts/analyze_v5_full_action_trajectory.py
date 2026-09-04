from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any


def mean(values: list[bool | float]) -> float:
    if not values:
        raise ValueError("mean requires values")
    return sum(map(float, values)) / len(values)


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


def _binomial_cdf(k: int, n: int, probability: float) -> float:
    return sum(
        math.comb(n, index)
        * probability**index
        * (1.0 - probability) ** (n - index)
        for index in range(k + 1)
    )


def exact_binomial_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if not 0 <= k <= n or n <= 0:
        raise ValueError("invalid binomial count")
    if k == 0:
        lower = 0.0
    else:
        lo, hi = 0.0, 1.0
        target = 1.0 - alpha / 2
        for _ in range(80):
            mid = (lo + hi) / 2
            if _binomial_cdf(k - 1, n, mid) > target:
                lo = mid
            else:
                hi = mid
        lower = (lo + hi) / 2
    if k == n:
        upper = 1.0
    else:
        lo, hi = 0.0, 1.0
        target = alpha / 2
        for _ in range(80):
            mid = (lo + hi) / 2
            if _binomial_cdf(k, n, mid) > target:
                lo = mid
            else:
                hi = mid
        upper = (lo + hi) / 2
    return lower, upper


def _cell(row: dict[str, Any], surface: str) -> bool:
    return (
        row["frame"] == "strategic"
        and row["incentive"] == "opposed"
        and row["surface_kind"] == surface
        and row["history"] == "redundant"
        and row["mapping_format"] == "prose"
    )


def _paired_cluster_gain(rows: list[dict[str, Any]]) -> tuple[list[Fraction], float]:
    clusters: dict[str, list[Fraction]] = defaultdict(list)
    for row in rows:
        direct = int(row["action_correct"])
        self_generated = int(row["trajectory_arms"]["self_generated"]["action_correct"])
        clusters[row["base_game_id"]].append(Fraction(self_generated - direct))
    values = [
        sum(parts, Fraction(0)) / len(parts)
        for _, parts in sorted(clusters.items())
    ]
    return values, exact_sign_flip_p(values)


def _action_accuracy(rows: list[dict[str, Any]], arm: str) -> float:
    if arm == "direct":
        return mean([row["action_correct"] for row in rows])
    return mean([row["trajectory_arms"][arm]["action_correct"] for row in rows])


def _report_accuracy(rows: list[dict[str, Any]]) -> float:
    return mean(
        [report["correct"] for row in rows for report in row["self_reports"].values()]
    )


def analyze(payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    rows = payload["rows"]
    primary = [row for row in rows if _cell(row, "assertion")]
    opaque = [row for row in rows if _cell(row, "opaque_token")]
    both_correct = [
        row for row in primary if all(r["correct"] for r in row["self_reports"].values())
    ]
    dissociations = [
        row
        for row in both_correct
        if not row["trajectory_arms"]["self_generated"]["action_correct"]
    ]
    dissociation_rate = len(dissociations) / len(primary)
    dissociation_ci = exact_binomial_interval(len(dissociations), len(primary))
    primary_report_accuracy = _report_accuracy(primary)
    persistent = (
        primary_report_accuracy
        >= float(config["primary_endpoints"]["self_report_accuracy_minimum"])
        and dissociation_rate
        >= float(config["primary_endpoints"]["persistent_wrong_action_rate_minimum"])
    )

    direct_accuracy = _action_accuracy(primary, "direct")
    self_accuracy = _action_accuracy(primary, "self_generated")
    oracle_accuracy = _action_accuracy(primary, "oracle_replay")
    swapped_accuracy = _action_accuracy(primary, "swapped_replay")
    rescue_gain = self_accuracy - direct_accuracy
    rescue_values, rescue_p = _paired_cluster_gain(primary)
    rescue = (
        rescue_gain
        >= float(
            config["secondary_endpoints"]["causal_rehearsal_rescue"]["minimum_gain"]
        )
        and rescue_p
        < float(config["secondary_endpoints"]["causal_rehearsal_rescue"]["alpha"])
    )

    order_accuracy = {
        "A_then_B": _action_accuracy(
            [row for row in primary if row["report_order"] == ["A", "B"]],
            "self_generated",
        ),
        "B_then_A": _action_accuracy(
            [row for row in primary if row["report_order"] == ["B", "A"]],
            "self_generated",
        ),
    }
    order_gap = order_accuracy["A_then_B"] - order_accuracy["B_then_A"]
    order_robust = abs(order_gap) <= float(
        config["secondary_endpoints"]["report_order_robustness"][
            "maximum_absolute_action_accuracy_gap"
        ]
    )
    opaque_accuracy = _action_accuracy(opaque, "self_generated")
    opaque_non_damage = opaque_accuracy >= float(
        config["secondary_endpoints"]["opaque_non_damage"][
            "minimum_self_generated_action_accuracy"
        ]
    )

    all_assertions = [row for row in rows if row["surface_kind"] == "assertion"]
    cell_breakdown = {}
    for frame in ("strategic", "device"):
        for history in ("none", "redundant"):
            for mapping in ("prose", "table"):
                cell = [
                    row
                    for row in all_assertions
                    if row["frame"] == frame
                    and row["history"] == history
                    and row["mapping_format"] == mapping
                ]
                cell_breakdown[f"{frame}:{history}:{mapping}"] = {
                    "n": len(cell),
                    "self_report_accuracy": _report_accuracy(cell),
                    "direct_action_accuracy": _action_accuracy(cell, "direct"),
                    "self_generated_action_accuracy": _action_accuracy(
                        cell, "self_generated"
                    ),
                    "oracle_replay_action_accuracy": _action_accuracy(
                        cell, "oracle_replay"
                    ),
                    "swapped_replay_action_accuracy": _action_accuracy(
                        cell, "swapped_replay"
                    ),
                }

    formatting = []
    for row in rows:
        formatting.extend(
            report["result"]["formatting_compliant"]
            for report in row["self_reports"].values()
        )
        formatting.extend(
            arm["result"]["formatting_compliant"]
            for arm in row["trajectory_arms"].values()
        )

    return {
        "schema_version": 1,
        "study_id": "V5-RBG-6",
        "source_run_id": payload["metadata"]["run_id"],
        "n": len(rows),
        "primary_full_trajectory_dissociation": {
            "population": "strategic opposed assertion, redundant history, prose mapping",
            "n": len(primary),
            "self_report_accuracy": primary_report_accuracy,
            "both_self_reports_correct": len(both_correct),
            "both_self_reports_correct_and_action_wrong": len(dissociations),
            "rate_over_all_trajectories": dissociation_rate,
            "exact_95_percent_binomial_interval": list(dissociation_ci),
            "persistent_signature": persistent,
        },
        "secondary_causal_rehearsal_rescue": {
            "direct_action_accuracy": direct_accuracy,
            "self_generated_action_accuracy": self_accuracy,
            "gain": rescue_gain,
            "two_sided_exact_base_game_sign_flip_p": rescue_p,
            "cluster_values": [float(value) for value in rescue_values],
            "rescue_signature": rescue,
        },
        "secondary_transcript_controls": {
            "oracle_replay_action_accuracy": oracle_accuracy,
            "swapped_replay_action_accuracy": swapped_accuracy,
            "oracle_minus_swapped_gap": oracle_accuracy - swapped_accuracy,
            "swapped_report_implied_action_accuracy": 1.0 - swapped_accuracy,
        },
        "secondary_report_order": {
            "self_generated_action_accuracy": order_accuracy,
            "A_then_B_minus_B_then_A": order_gap,
            "robust": order_robust,
        },
        "secondary_opaque_non_damage": {
            "n": len(opaque),
            "self_report_accuracy": _report_accuracy(opaque),
            "self_generated_action_accuracy": opaque_accuracy,
            "passed": opaque_non_damage,
        },
        "diagnostics": {
            "opposed_assertion_cells": cell_breakdown,
            "forced_choice_formatting_compliance": mean(formatting),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = analyze(payload, config)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite analysis: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
