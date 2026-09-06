"""CPU join of prepared mid-trajectory payload + GHA raw scores → arm metrics.

Recomputes base-equal choice1/choice2 accuracy, persistence, flip, intervention
contrasts, and a *descriptive* opaque/prose asymmetry section. Does not call
Modal or any GPU. Immutable: fails if the analysis path already exists (no unlink).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from jspace_policy.sprint_runtime import write_new

EXPECTED_PAYLOAD_SHA256 = (
    "50b854ae18ab584a25234e62687de4d4f6de8c467d5656ec9dda89a0b1d7a5d6"
)
RUN_ID = "ask-mid-traj-qwen38-n16-v1"
ACTIONS_RUN_URL = (
    "https://github.com/thewildofficial/when-words-override-consequences"
    "/actions/runs/34052527423"
)
ACTIONS_RUN_ID = 34052527423
N_SCORES = 864
N_RECORDS = 512
ASK_FIRST_SWAPPED_SURFACE_NOTE = (
    "Opposite direction from ask-first swapped (C15/C16), where prose failed "
    "more than opaque (prose accuracy 0.71875 vs opaque 0.9375 on gha-report16)."
)


def _score_rows(
    payload: dict[str, Any], scores: dict[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    queries = payload["queries"]
    for rec in payload["records"]:
        arm = str(rec["arm"])
        choice1 = scores[rec["choice1"]]["choice"]
        c1_len = int(queries[rec["choice1"]]["length"])
        if arm == "mid_ask_self":
            mid = scores[rec["mid"][choice1]]["choice"]
            choice2 = scores[rec["choice2"][choice1 + mid]]["choice"]
            c2_key = rec["choice2"][choice1 + mid]
        else:
            mid = str(rec["arm_mid_token"])
            choice2 = scores[rec["choice2"][choice1]]["choice"]
            c2_key = rec["choice2"][choice1]
        c2_len = int(queries[c2_key]["length"])
        expected = rec["expected_action"]
        flip = choice2 != choice1
        rows.append(
            {
                "base_id": rec["base_id"],
                "split": rec["split"],
                "arm": arm,
                "frame": rec["frame"],
                "surface_kind": rec["surface_kind"],
                "policy": rec["policy"],
                "expected_action": expected,
                "expected_mid_token": rec["expected_mid_token"],
                "choice1": choice1,
                "choice2": choice2,
                "mid": mid,
                "choice1_correct": float(choice1 == expected),
                "choice2_correct": float(choice2 == expected),
                "persist": float(choice2 == choice1),
                "flip": float(flip),
                "mid_correct": float(mid == rec["expected_mid_token"]),
                "flip_to_wrong": float(flip and choice2 != expected),
                "persist_correct": float((not flip) and choice2 == expected),
                "c1_to_c2_length_delta": c2_len - c1_len,
            }
        )
    return rows


def _base_equal(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    by_base: dict[Any, list[float]] = defaultdict(list)
    for row in rows:
        by_base[row["base_id"]].append(float(row[metric]))
    means = [sum(values) / len(values) for values in by_base.values()]
    micro = (sum(float(row[metric]) for row in rows) / len(rows)) if rows else float("nan")
    return {
        "n_rows": len(rows),
        "n_bases": len(by_base),
        "base_equal": (sum(means) / len(means)) if means else float("nan"),
        "micro": micro,
    }


def _arm_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for metric in (
        "choice1_correct",
        "choice2_correct",
        "persist",
        "flip",
        "mid_correct",
    ):
        detail = _base_equal(rows, metric)
        out[metric] = detail["base_equal"]
        out[f"{metric}_detail"] = detail
    return out


def _policy_short(policy: str) -> str:
    if "opposed" in policy:
        return "opposed"
    if "direct" in policy:
        return "direct"
    return policy


def _swapped_flip_cells(swapped: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cells: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in swapped:
        key = (
            str(row["surface_kind"]),
            _policy_short(str(row["policy"])),
            str(row["frame"]),
        )
        cells[key].append(row)
    out: list[dict[str, Any]] = []
    for surface, policy, frame in sorted(cells):
        items = cells[(surface, policy, frame)]
        n = len(items)
        n_flip = int(sum(float(row["flip"]) for row in items))
        out.append(
            {
                "surface_kind": surface,
                "policy_kind": policy,
                "frame": frame,
                "n": n,
                "n_flip": n_flip,
                "flip": (n_flip / n) if n else float("nan"),
                "choice2_accuracy": (
                    sum(float(row["choice2_correct"]) for row in items) / n
                    if n
                    else float("nan")
                ),
            }
        )
    return out


def _opaque_prose_asymmetry(swapped: list[dict[str, Any]]) -> dict[str, Any]:
    """DESCRIPTIVE ONLY — not a locked interaction claim."""

    by_surface: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in swapped:
        by_surface[str(row["surface_kind"])].append(row)

    opaque_flip = _base_equal(by_surface["opaque"], "flip")["base_equal"]
    prose_flip = _base_equal(by_surface["prose"], "flip")["base_equal"]

    opaque_by_base: dict[Any, list[float]] = defaultdict(list)
    prose_by_base: dict[Any, list[float]] = defaultdict(list)
    for row in swapped:
        bucket = opaque_by_base if row["surface_kind"] == "opaque" else prose_by_base
        bucket[row["base_id"]].append(float(row["flip"]))

    opaque_gt = prose_gt = ties = 0
    for base_id in sorted(opaque_by_base):
        o = sum(opaque_by_base[base_id]) / len(opaque_by_base[base_id])
        p = sum(prose_by_base[base_id]) / len(prose_by_base[base_id])
        if o > p:
            opaque_gt += 1
        elif p > o:
            prose_gt += 1
        else:
            ties += 1

    # Within base × policy × frame pairs (n=64): opaque-only vs prose-only flips.
    pairs: dict[tuple[Any, ...], dict[str, float]] = defaultdict(dict)
    for row in swapped:
        key = (row["base_id"], row["policy"], row["frame"])
        pairs[key][str(row["surface_kind"])] = float(row["flip"])
    opaque_only = prose_only = 0
    for surfaces in pairs.values():
        if surfaces.get("opaque") == 1.0 and surfaces.get("prose") == 0.0:
            opaque_only += 1
        if surfaces.get("prose") == 1.0 and surfaces.get("opaque") == 0.0:
            prose_only += 1

    length_by_surface: dict[str, list[float]] = defaultdict(list)
    for row in swapped:
        length_by_surface[str(row["surface_kind"])].append(
            float(row["c1_to_c2_length_delta"])
        )

    def _mean(values: list[float]) -> float:
        return (sum(values) / len(values)) if values else float("nan")

    return {
        "standing": "descriptive_only_not_locked_interaction",
        "base_equal_flip": {
            "opaque": opaque_flip,
            "prose": prose_flip,
            "opaque_minus_prose": opaque_flip - prose_flip,
        },
        "across_16_bases": {
            "opaque_gt_prose": opaque_gt,
            "prose_gt_opaque": prose_gt,
            "ties": ties,
        },
        "within_base_policy_frame_pairs": {
            "n_pairs": len(pairs),
            "opaque_only_flips": opaque_only,
            "prose_only_flips": prose_only,
        },
        "length_c1_to_c2_delta_mean": {
            "prose": _mean(length_by_surface["prose"]),
            "opaque": _mean(length_by_surface["opaque"]),
            "note": "Not an obvious length confound (means nearly equal).",
        },
        "direction_vs_ask_first_swapped": ASK_FIRST_SWAPPED_SURFACE_NOTE,
        "limits": [
            "Descriptive split within one discovery pilot payload.",
            "Not a locked surface-main-effect or interaction claim.",
            "Opposite direction from ask-first swapped prose>opaque failure (C15/C16).",
        ],
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
    if payload.get("protocol_id") != "ask_mid_trajectory":
        raise ValueError("expected protocol_id=ask_mid_trajectory")
    if payload.get("task") != "ask_mid_trajectory":
        raise ValueError("expected task=ask_mid_trajectory")
    scores = raw["scores"]
    if len(scores) != N_SCORES:
        raise ValueError(f"expected {N_SCORES} scores, got {len(scores)}")
    if len(payload["records"]) != N_RECORDS:
        raise ValueError(f"expected {N_RECORDS} records, got {len(payload['records'])}")

    rows = _score_rows(payload, scores)
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_arm[str(row["arm"])].append(row)

    arm_metrics = {arm: _arm_metrics(items) for arm, items in sorted(by_arm.items())}
    swapped = by_arm["mid_ask_swapped"]
    control = by_arm["mid_no_ask_control"]
    self_ask = by_arm["mid_ask_self"]

    intervention = {
        "treatment": "mid_ask_self",
        "control": "mid_no_ask_control",
        "flip": (
            arm_metrics["mid_ask_self"]["flip"]
            - arm_metrics["mid_no_ask_control"]["flip"]
        ),
        "choice2_accuracy": (
            arm_metrics["mid_ask_self"]["choice2_correct"]
            - arm_metrics["mid_no_ask_control"]["choice2_correct"]
        ),
        "interpretation": (
            "Zero contrast → self mid-ask is not an intervention on this pilot; "
            "matched non-consequence chatter and consequence self-ask leave "
            "choice2 identical (ceiling persistence)."
        ),
    }

    swapped_story = {
        "choice1_accuracy": arm_metrics["mid_ask_swapped"]["choice1_correct"],
        "choice2_accuracy": arm_metrics["mid_ask_swapped"]["choice2_correct"],
        "persist": arm_metrics["mid_ask_swapped"]["persist"],
        "flip": arm_metrics["mid_ask_swapped"]["flip"],
        "every_flip_lands_on_wrong_action": all(
            float(row["flip"]) == 0.0 or float(row["flip_to_wrong"]) == 1.0
            for row in swapped
        ),
        "every_persist_stays_correct": all(
            float(row["persist"]) == 0.0 or float(row["persist_correct"]) == 1.0
            for row in swapped
        ),
        "n_flips": int(sum(float(row["flip"]) for row in swapped)),
        "n_persists": int(sum(float(row["persist"]) for row in swapped)),
        "flip_cells": _swapped_flip_cells(swapped),
        "flip_by_frame": {
            frame: _base_equal(
                [row for row in swapped if row["frame"] == frame], "flip"
            )["base_equal"]
            for frame in ("strategic", "nonagentic")
        },
    }

    return {
        "run_id": RUN_ID,
        "actions_run_id": ACTIONS_RUN_ID,
        "actions_run_url": ACTIONS_RUN_URL,
        "stage": "baseline",
        "task": "ask_mid_trajectory",
        "protocol_id": "ask_mid_trajectory",
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
        "story_lead": (
            "Asking mid-trajectory does nothing on this pilot: self mid-ask vs "
            "no-ask control leaves flip and choice2 accuracy unchanged at 0.0. "
            "A lied mid-answer rewrites ~60% of second presses "
            f"(swapped flip={arm_metrics['mid_ask_swapped']['flip']})."
        ),
        "arm_metrics_base_equal": {
            arm: {
                "choice1_accuracy": metrics["choice1_correct"],
                "choice2_accuracy": metrics["choice2_correct"],
                "persist": metrics["persist"],
                "flip": metrics["flip"],
                "mid_correct": metrics["mid_correct"],
            }
            for arm, metrics in arm_metrics.items()
        },
        "arm_metric_detail": arm_metrics,
        "intervention_contrast_mid_ask_self_minus_control": intervention,
        "swapped_mid_content": swapped_story,
        "opaque_prose_asymmetry_descriptive": _opaque_prose_asymmetry(swapped),
        "ceiling_arms": {
            arm: {
                "choice1_accuracy": arm_metrics[arm]["choice1_correct"],
                "choice2_accuracy": arm_metrics[arm]["choice2_correct"],
                "persist": arm_metrics[arm]["persist"],
                "flip": arm_metrics[arm]["flip"],
            }
            for arm in ("mid_no_ask_control", "mid_ask_self", "mid_ask_oracle")
        },
        "mid_ask_self_mid_answers_perfect": arm_metrics["mid_ask_self"]["mid_correct"]
        == 1.0,
        "n_control_rows": len(control),
        "n_self_rows": len(self_ask),
        "interpretation_limits": [
            "Discovery pilot / engineering_pilot; not locked confirmation.",
            "Self mid-ask is not an intervention here (contrast 0.0); content "
            "sensitivity without ask-as-intervention matches stop rule 2.",
            "Swapped steering: every flip → wrong action; every persist → correct.",
            "Opaque>prose flip asymmetry is DESCRIPTIVE ONLY — not a locked interaction.",
            "Opposite surface direction from ask-first swapped (C15/C16).",
            "GHA ledger: ask-mid-traj-qwen38-n16-v1 retained (~$0.7829); "
            "historical reservations.jsonl unchanged.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepared",
        type=Path,
        default=Path(
            "results/report_reactivity/ask-mid-traj-qwen38-n16-v1/prepared.json"
        ),
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("results/report_reactivity/ask-mid-traj-qwen38-n16-v1/raw.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/report_reactivity/ask-mid-traj-qwen38-n16-v1/"
            "analysis/arm_accuracy_summary.json"
        ),
    )
    args = parser.parse_args()
    payload = json.loads(args.prepared.read_text())
    raw = json.loads(args.raw.read_text())
    summary = summarize(payload, raw)
    # Fail closed: never unlink/overwrite an existing analysis artifact.
    write_new(args.output, summary)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "story_lead": summary["story_lead"],
                "arm_metrics_base_equal": summary["arm_metrics_base_equal"],
                "intervention_contrast": {
                    "flip": summary["intervention_contrast_mid_ask_self_minus_control"][
                        "flip"
                    ],
                    "choice2_accuracy": summary[
                        "intervention_contrast_mid_ask_self_minus_control"
                    ]["choice2_accuracy"],
                },
                "opaque_prose_asymmetry_descriptive": {
                    "opaque_minus_prose_flip": summary[
                        "opaque_prose_asymmetry_descriptive"
                    ]["base_equal_flip"]["opaque_minus_prose"],
                    "standing": summary["opaque_prose_asymmetry_descriptive"][
                        "standing"
                    ],
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
