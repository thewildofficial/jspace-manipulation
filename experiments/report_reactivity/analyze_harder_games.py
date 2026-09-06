"""CPU join of prepared harder-games payload + GHA raw scores → arm accuracy.

Recomputes base-equal action accuracies, primary contrasts, and swapped cell
map with ``jspace_policy.sprint_analysis``. Does not call Modal or any GPU.
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
    "a8b43df94451398b4edfecddb1a3f1c821c014e1d4212d6cc207be2dd077964c"
)
RUN_ID = "harder-games-qwen38-n16-v1"
ACTIONS_RUN_URL = (
    "https://github.com/thewildofficial/when-words-override-consequences"
    "/actions/runs/34051102729"
)
HISTORY_MODE = "redundant"
PRIOR_MINIMAL_RUN_ID = "gha-report16-38-v1"
PRIOR_ACTIONS_RUN_URL = (
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


def _swapped_cell_map(swapped: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cells: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in swapped:
        key = (str(row["surface_kind"]), str(row["policy"]), str(row["frame"]))
        cells[key].append(row)
    out: list[dict[str, Any]] = []
    for surface, policy, frame in sorted(cells):
        items = cells[(surface, policy, frame)]
        n = len(items)
        n_correct = int(sum(float(row["correct"]) for row in items))
        out.append(
            {
                "surface_kind": surface,
                "policy_kind": policy,
                "frame": frame,
                "n": n,
                "n_correct": n_correct,
                "n_failures": n - n_correct,
                "accuracy": (n_correct / n) if n else float("nan"),
            }
        )
    return out


def summarize(payload: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    if payload.get("sha256") != EXPECTED_PAYLOAD_SHA256:
        raise ValueError(
            f"prepared sha256 {payload.get('sha256')!r} != {EXPECTED_PAYLOAD_SHA256}"
        )
    if raw.get("payload_sha256") != EXPECTED_PAYLOAD_SHA256:
        raise ValueError(
            f"raw payload_sha256 {raw.get('payload_sha256')!r} != {EXPECTED_PAYLOAD_SHA256}"
        )
    if payload.get("history_mode") != HISTORY_MODE:
        raise ValueError(
            f"expected history_mode={HISTORY_MODE!r}, got {payload.get('history_mode')!r}"
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
    swapped_cells = _swapped_cell_map(swapped)

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
        "history_mode": HISTORY_MODE,
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
        "swapped_cell_map": swapped_cells,
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
        "contrast_vs_minimal_history": {
            "prior_run_id": PRIOR_MINIMAL_RUN_ID,
            "prior_actions_run_url": PRIOR_ACTIONS_RUN_URL,
            "note": (
                "Prior ask-first minimal-history baseline also had Direct=1.0; "
                "swapped≈0.828 with worst cell prose/opposed/strategic=0.375 (C14–C16). "
                "Redundant demos did not break Direct ceiling."
            ),
        },
        "interpretation_limits": [
            "Discovery pilot / engineering_pilot; not locked confirmation.",
            "Substantive null: redundant correct demos did NOT break Direct ceiling "
            "(still 1.0) → ask-first rescue remains unidentifiable this way.",
            "Not claimed: that demos make games harder in a useful way.",
            "Swapped still leaks under redundant history (~0.789); strategic cells worst; "
            "worst cell remains prose+opposed+strategic (now 0.5 vs prior 0.375).",
            "GHA ledger: reservations_gha includes harder-games-qwen38-n16-v1 "
            "(~$0.7829 ceiling); historical reservations.jsonl unchanged.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepared",
        type=Path,
        default=Path("artifacts/prepared/harder-games-qwen38-n16-v1.json"),
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("results/report_reactivity/harder-games-qwen38-n16-v1/raw.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/report_reactivity/harder-games-qwen38-n16-v1/"
            "analysis/arm_accuracy_summary.json"
        ),
    )
    args = parser.parse_args()
    payload = json.loads(args.prepared.read_text())
    raw = json.loads(args.raw.read_text())
    summary = summarize(payload, raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()
    write_new(args.output, summary)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "history_mode": summary["history_mode"],
                "arm_accuracies_base_equal": summary["arm_accuracies_base_equal"],
                "primary_contrasts": {
                    key: value["effect"]
                    for key, value in summary["primary_contrasts"].items()
                },
                "swapped_cell_map": {
                    f"{c['surface_kind']}|{c['policy_kind']}|{c['frame']}": c["accuracy"]
                    for c in summary["swapped_cell_map"]
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
