"""CPU join of prepared report payload + GHA raw scores → arm accuracy summary.

Recomputes base-equal action accuracies and primary contrasts with
``jspace_policy.sprint_analysis``. Does not call Modal or any GPU.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from jspace_policy.sprint_analysis import grouped_paired_analysis, primary_gate
from jspace_policy.sprint_runtime import write_new

EXPECTED_PAYLOAD_SHA256 = (
    "126ea05173558cd161f017922a936c8248704d2ac35dd34c3213b1de07bf257d"
)
RUN_ID = "gha-report16-38-v1"
ACTIONS_RUN_URL = (
    "https://github.com/thewildofficial/when-words-override-consequences"
    "/actions/runs/34048123330"
)


def _score_rows(payload: dict[str, Any], scores: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in payload["records"]:
        arm = str(rec["arm"])
        base = {
            "base_id": rec["base_id"],
            "split": rec["split"],
            "arm": arm,
            "frame": rec["frame"],
            "surface_kind": rec["surface_kind"],
            "policy": rec["policy"],
            "expected_action": rec["expected_action"],
        }
        if arm == "self_report":
            r1 = scores[rec["report1"]]["choice"]
            r2 = scores[rec["report2"][r1]]["choice"]
            choice = scores[rec["actions"][r1 + r2]]["choice"]
            reports_correct = int(
                r1 == rec["expected_reports"][0] and r2 == rec["expected_reports"][1]
            )
            rows.append(
                {
                    **base,
                    "choice": choice,
                    "correct": float(choice == rec["expected_action"]),
                    "reports_correct": reports_correct,
                    "generated_reports": [r1, r2],
                }
            )
        else:
            choice = scores[rec["action"]]["choice"]
            rows.append(
                {
                    **base,
                    "choice": choice,
                    "correct": float(choice == rec["expected_action"]),
                }
            )
    return rows


def _base_equal_accuracy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_base: dict[Any, list[float]] = defaultdict(list)
    for row in rows:
        by_base[row["base_id"]].append(float(row["correct"]))
    means = [sum(values) / len(values) for values in by_base.values()]
    n_correct = int(sum(float(row["correct"]) for row in rows))
    return {
        "n_rows": len(rows),
        "n_bases": len(by_base),
        "n_correct": n_correct,
        "base_equal_accuracy": (sum(means) / len(means)) if means else float("nan"),
        "micro_accuracy": (n_correct / len(rows)) if rows else float("nan"),
    }


def summarize(payload: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    if payload.get("sha256") != EXPECTED_PAYLOAD_SHA256:
        raise ValueError(
            f"prepared sha256 {payload.get('sha256')!r} != {EXPECTED_PAYLOAD_SHA256}"
        )
    if raw.get("payload_sha256") != EXPECTED_PAYLOAD_SHA256:
        raise ValueError(
            f"raw payload_sha256 {raw.get('payload_sha256')!r} != {EXPECTED_PAYLOAD_SHA256}"
        )
    scores = raw["scores"]
    if len(scores) != 960:
        raise ValueError(f"expected 960 scores, got {len(scores)}")
    if len(payload["records"]) != 768:
        raise ValueError(f"expected 768 records, got {len(payload['records'])}")

    rows = _score_rows(payload, scores)
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_arm[str(row["arm"])].append(row)

    arm_accuracies = {
        arm: _base_equal_accuracy(items) for arm, items in sorted(by_arm.items())
    }
    swapped = by_arm["swapped"]
    swapped_by_surface = {
        surface: _base_equal_accuracy(
            [row for row in swapped if row["surface_kind"] == surface]
        )
        for surface in ("prose", "opaque")
    }

    analysis_rows = [
        {
            "base_id": row["base_id"],
            "split": row["split"],
            "arm": row["arm"],
            "correct": float(row["correct"]),
        }
        for row in rows
    ]
    contrasts: dict[str, Any] = {}
    for treatment, control in (
        ("self_report", "matched_control"),
        ("self_report", "direct"),
    ):
        paired = grouped_paired_analysis(
            analysis_rows,
            treatment_arm=treatment,
            control_arm=control,
            split="discovery",
            bootstrap_resamples=5000,
            seed=1729,
        )
        contrasts[f"{treatment}_minus_{control}"] = {
            "effect": paired["effect"],
            "ci_low": paired["ci_low"],
            "ci_high": paired["ci_high"],
            "p_value": paired["p_value"],
            "n_bases": paired["n_bases"],
        }

    gate = primary_gate(
        analysis_rows, split="discovery", bootstrap_resamples=5000, seed=1729
    )
    self_rows = by_arm["self_report"]
    return {
        "run_id": RUN_ID,
        "actions_run_url": ACTIONS_RUN_URL,
        "stage": "baseline",
        "task": "report",
        "split": "discovery",
        "bases": 16,
        "batch_size": 4,
        "model_id": raw.get("model_id"),
        "revision": raw.get("revision"),
        "payload_sha256": EXPECTED_PAYLOAD_SHA256,
        "status": raw.get("status"),
        "parity": raw.get("parity"),
        "gpu": raw.get("gpu"),
        "elapsed_seconds": raw.get("elapsed_seconds"),
        "load_seconds": raw.get("load_seconds"),
        "n_scores": len(scores),
        "n_records": len(payload["records"]),
        "arm_accuracies_base_equal": {
            arm: value["base_equal_accuracy"] for arm, value in arm_accuracies.items()
        },
        "arm_accuracy_detail": arm_accuracies,
        "swapped_by_surface_kind": swapped_by_surface,
        "self_report_both_reports_correct": {
            "n_correct": int(sum(int(row["reports_correct"]) for row in self_rows)),
            "n_rows": len(self_rows),
            "chaining": "report1→report2[r1]→actions[r1r2]",
        },
        "primary_contrasts": contrasts,
        "primary_gate": {
            key: gate[key]
            for key in (
                "effect",
                "ci_low",
                "ci_high",
                "p_value",
                "n_bases",
                "gate_pass",
                "decision",
                "status",
                "confirmatory",
                "selection_allowed",
                "reason",
            )
        },
        "interpretation_limits": [
            "Discovery pilot / engineering_pilot; not locked confirmation; not a novelty/priority claim.",
            "Ceiling on Direct/self/control means rewrite-vs-reveal rescue is unidentifiable here (stop rule for mechanistic rescue / reports-fix-failures).",
            "Swapped action accuracy below ceiling shows report-content can steer action; drop is larger on prose than opaque.",
            "Replicates qualitative pattern of earlier claim C2/C3 on a fresh GHA path + nonce corpus; do not overclaim independence beyond what's justified.",
            "GHA ledger: reservations_gha includes gha-report16-38-v1 (~$0.7829 ceiling); total with prior v2 preflight retention — see reservations_gha.jsonl.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepared",
        type=Path,
        default=Path(
            "results/report_reactivity/gha-report16-38-v1/prepared.json"
        ),
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("results/report_reactivity/gha-report16-38-v1/raw.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/report_reactivity/gha-report16-38-v1/analysis/arm_accuracy_summary.json"
        ),
    )
    args = parser.parse_args()
    payload = json.loads(args.prepared.read_text())
    raw = json.loads(args.raw.read_text())
    summary = summarize(payload, raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_new(args.output, summary)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "arm_accuracies_base_equal": summary["arm_accuracies_base_equal"],
                "primary_contrasts": {
                    key: value["effect"]
                    for key, value in summary["primary_contrasts"].items()
                },
                "swapped_by_surface_kind": {
                    key: value["base_equal_accuracy"]
                    for key, value in summary["swapped_by_surface_kind"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
