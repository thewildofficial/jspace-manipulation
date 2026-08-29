from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Callable
from fractions import Fraction
from typing import Any


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def content_hash_valid(payload: dict[str, Any]) -> bool:
    claimed = payload.get("content_sha256")
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    return claimed == canonical_sha256(body)


def exact_sign_flip_p(values: list[Fraction | float]) -> float:
    fractions = [
        value if isinstance(value, Fraction) else Fraction(str(value)) for value in values
    ]
    nonzero = [value for value in fractions if value]
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


def _mean(values: list[bool | float]) -> float:
    if not values:
        raise ValueError("mean requires at least one value")
    return sum(map(float, values)) / len(values)


def report_accuracy(rows: list[dict[str, Any]]) -> float:
    return _mean(
        [report["correct"] for row in rows for report in row["option_reports"].values()]
    )


def _cell(
    rows: list[dict[str, Any]],
    *,
    incentive: str,
    surface: str,
    history: str,
    mapping: str,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["incentive"] == incentive
        and row["surface_kind"] == surface
        and row["history"] == history
        and row["mapping_format"] == mapping
    ]


def _cluster_contrast(
    rows: list[dict[str, Any]],
    predicates: list[tuple[Callable[[dict[str, Any]], bool], int]],
) -> tuple[list[Fraction], float]:
    clusters: dict[str, list[list[bool]]] = defaultdict(lambda: [[] for _ in predicates])
    for row in rows:
        for index, (predicate, _) in enumerate(predicates):
            if predicate(row):
                clusters[row["base_game_id"]][index].append(bool(row["action_correct"]))
    values: list[Fraction] = []
    for base_id, cells in clusters.items():
        if any(not cell for cell in cells):
            raise ValueError(f"incomplete contrast for base {base_id}")
        value = Fraction(0)
        for cell, (_, sign) in zip(cells, predicates, strict=True):
            value += sign * Fraction(sum(cell), len(cell))
        values.append(value)
    return values, exact_sign_flip_p(values)


def eligible_recipients(
    rows: list[dict[str, Any]], split: str, donor_family: str | None = None
) -> list[dict[str, Any]]:
    candidates = _cell(
        rows,
        incentive="opposed",
        surface="assertion",
        history="redundant",
        mapping="prose",
    )
    output = []
    for row in candidates:
        if row["split"] != split or row["action_correct"]:
            continue
        if not all(report["correct"] for report in row["option_reports"].values()):
            continue
        matched = [
            other
            for other in rows
            if other["base_game_id"] == row["base_game_id"]
            and other["frame"] == row["frame"]
            and other["incentive"] == row["incentive"]
            and other["history"] == "redundant"
            and (
                (other["surface_kind"] == "assertion" and other["mapping_format"] == "table")
                or (
                    other["surface_kind"] == "opaque_token"
                    and other["mapping_format"] == "prose"
                )
            )
        ]
        if donor_family == "table":
            matched = [row for row in matched if row["mapping_format"] == "table"]
        elif donor_family == "opaque":
            matched = [row for row in matched if row["surface_kind"] == "opaque_token"]
        elif donor_family is not None:
            raise ValueError(f"unknown donor family: {donor_family}")
        if any(other["action_correct"] for other in matched):
            output.append(row)
    return output


def analyze_behavior(payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    rows = payload["rows"]
    gates = config["behavior_gate"]
    all_reports = report_accuracy(rows)
    opaque = [row for row in rows if row["surface_kind"] == "opaque_token"]
    opaque_accuracy = _mean([row["action_correct"] for row in opaque])

    prose_none = _cell(
        rows, incentive="opposed", surface="assertion", history="none", mapping="prose"
    )
    prose_history = _cell(
        rows,
        incentive="opposed",
        surface="assertion",
        history="redundant",
        mapping="prose",
    )
    prose_harm = _mean([r["action_correct"] for r in prose_none]) - _mean(
        [r["action_correct"] for r in prose_history]
    )
    prose_report_gap = report_accuracy(prose_none) - report_accuracy(prose_history)
    prose_values, prose_p = _cluster_contrast(
        rows,
        [
            (
                lambda r: r["incentive"] == "opposed"
                and r["surface_kind"] == "assertion"
                and r["mapping_format"] == "prose"
                and r["history"] == "none",
                1,
            ),
            (
                lambda r: r["incentive"] == "opposed"
                and r["surface_kind"] == "assertion"
                and r["mapping_format"] == "prose"
                and r["history"] == "redundant",
                -1,
            ),
        ],
    )
    opaque_none = _cell(
        rows,
        incentive="opposed",
        surface="opaque_token",
        history="none",
        mapping="prose",
    )
    opaque_history = _cell(
        rows,
        incentive="opposed",
        surface="opaque_token",
        history="redundant",
        mapping="prose",
    )
    opaque_harm = _mean([r["action_correct"] for r in opaque_none]) - _mean(
        [r["action_correct"] for r in opaque_history]
    )
    did = prose_harm - opaque_harm
    did_values, did_p = _cluster_contrast(
        rows,
        [
            (
                lambda r: r["incentive"] == "opposed"
                and r["surface_kind"] == "assertion"
                and r["mapping_format"] == "prose"
                and r["history"] == "none",
                1,
            ),
            (
                lambda r: r["incentive"] == "opposed"
                and r["surface_kind"] == "assertion"
                and r["mapping_format"] == "prose"
                and r["history"] == "redundant",
                -1,
            ),
            (
                lambda r: r["incentive"] == "opposed"
                and r["surface_kind"] == "opaque_token"
                and r["mapping_format"] == "prose"
                and r["history"] == "none",
                -1,
            ),
            (
                lambda r: r["incentive"] == "opposed"
                and r["surface_kind"] == "opaque_token"
                and r["mapping_format"] == "prose"
                and r["history"] == "redundant",
                1,
            ),
        ],
    )
    table_none = _cell(
        rows, incentive="opposed", surface="assertion", history="none", mapping="table"
    )
    table_history = _cell(
        rows,
        incentive="opposed",
        surface="assertion",
        history="redundant",
        mapping="table",
    )
    table_harm = _mean([r["action_correct"] for r in table_none]) - _mean(
        [r["action_correct"] for r in table_history]
    )
    locked_eligible_by_donor = {
        donor: eligible_recipients(rows, "locked", donor) for donor in ("table", "opaque")
    }
    locked_eligible = eligible_recipients(rows, "locked")
    passed = (
        all_reports >= float(gates["minimum_option_report_accuracy"])
        and opaque_accuracy >= float(gates["minimum_opaque_action_accuracy"])
        and prose_harm >= float(gates["minimum_prose_assertion_history_harm"])
        and abs(prose_report_gap) <= float(gates["maximum_absolute_report_gap"])
        and prose_p < float(gates["maximum_exact_cluster_p"])
        and did >= float(gates["minimum_assertion_minus_opaque_history_harm"])
        and did_p < float(gates["maximum_exact_cluster_p"])
        and table_harm <= float(gates["maximum_table_history_harm"])
        and min(map(len, locked_eligible_by_donor.values()))
        >= int(gates["minimum_locked_eligible_recipients"])
    )
    return {
        "schema_version": 1,
        "study_id": "V5-RBG-5",
        "source_run_id": payload["metadata"]["run_id"],
        "gate_pass": passed,
        "n_rows": len(rows),
        "all_option_report_accuracy": all_reports,
        "opaque_action_accuracy": opaque_accuracy,
        "prose_assertion_history_harm": prose_harm,
        "prose_option_report_gap": prose_report_gap,
        "prose_exact_cluster_p": prose_p,
        "prose_cluster_values": list(map(float, prose_values)),
        "opaque_history_harm": opaque_harm,
        "assertion_minus_opaque_history_harm": did,
        "semantic_exact_cluster_p": did_p,
        "semantic_cluster_values": list(map(float, did_values)),
        "table_history_harm": table_harm,
        "locked_eligible_recipients": len(locked_eligible),
        "locked_eligible_condition_ids": [row["condition_id"] for row in locked_eligible],
        "locked_eligible_by_donor": {
            donor: {
                "n": len(donor_rows),
                "condition_ids": [row["condition_id"] for row in donor_rows],
            }
            for donor, donor_rows in locked_eligible_by_donor.items()
        },
    }


def correct_action_margin(row: dict[str, Any], legal_logits: dict[str, float]) -> float:
    correct = str(row["expected_action"])
    incorrect = "B" if correct == "A" else "A"
    return float(legal_logits[correct]) - float(legal_logits[incorrect])


def normalized_recovery(
    destination_margin: float,
    source_margin: float,
    patched_margin: float,
    minimum_denominator: float,
) -> float | None:
    denominator = source_margin - destination_margin
    if abs(denominator) < minimum_denominator:
        return None
    return (patched_margin - destination_margin) / denominator


def select_patch_site(
    rows: list[dict[str, Any]], config: dict[str, Any], dataset_sha256: str
) -> dict[str, Any]:
    if not rows:
        raise ValueError("no discovery patch rows")
    anchor_rank = {
        anchor: index for index, anchor in enumerate(config["patch"]["candidate_anchors"])
    }
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["donor_family"], int(row["layer"]), row["anchor"])].append(row)
    summaries = []
    for (donor, layer, anchor), items in grouped.items():
        by_base: dict[str, list[float]] = defaultdict(list)
        for item in items:
            by_base[item["base_game_id"]].append(float(item["margin_change"]))
        cluster_values = [_mean(values) for values in by_base.values()]
        summaries.append(
            {
                "donor_family": donor,
                "layer": layer,
                "anchor": anchor,
                "n_rows": len(items),
                "n_bases": len(by_base),
                "cluster_mean_margin_change": _mean(cluster_values),
                "exact_cluster_sign_flip_p": exact_sign_flip_p(cluster_values),
            }
        )
    donor_rank = {"table": 0, "opaque": 1}
    summaries.sort(
        key=lambda row: (
            -row["cluster_mean_margin_change"],
            donor_rank[row["donor_family"]],
            row["layer"],
            anchor_rank[row["anchor"]],
        )
    )
    winner = summaries[0]
    status = (
        "frozen_after_discovery_before_locked_capture"
        if winner["cluster_mean_margin_change"] > 0
        else "no_positive_discovery_candidate_locked_phase_closed"
    )
    artifact = {
        "schema_version": 1,
        "study_id": "V5-RBG-5",
        "status": status,
        "dataset_sha256": dataset_sha256,
        "selection_metric": config["patch"]["selection_metric"],
        "selected": winner if winner["cluster_mean_margin_change"] > 0 else None,
        "selection_table": summaries,
    }
    artifact["content_sha256"] = canonical_sha256(artifact)
    return artifact


def analyze_locked_patches(
    patch_results: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    primary = patch_results["primary"]
    if not primary:
        raise ValueError("locked patch results contain no primary rows")
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in primary:
        grouped[row["base_game_id"]].append(float(row["margin_change"]))
    cluster_values = [_mean(values) for values in grouped.values()]
    repair_rate = _mean([row["repaired"] for row in primary])
    margin_change = _mean([row["margin_change"] for row in primary])
    exact_p = exact_sign_flip_p(cluster_values)
    primary_pass = (
        repair_rate >= float(config["patch"]["minimum_locked_repair_rate"])
        and margin_change > 0
        and exact_p < float(config["patch"]["maximum_exact_cluster_p"])
    )
    normalized = [
        float(row["normalized_recovery"])
        for row in primary
        if row["normalized_recovery"] is not None
    ]
    controls_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in patch_results["controls"]:
        controls_by_name[row["control"]].append(row)
    non_damage = {}
    maximum_loss = float(config["patch"]["maximum_control_accuracy_loss"])
    for name in ("non_damage_aligned", "non_damage_table", "non_damage_opaque"):
        rows = controls_by_name[name]
        before = _mean([row["correct_before"] for row in rows]) if rows else None
        after = _mean([row["correct_after"] for row in rows]) if rows else None
        non_damage[name] = {
            "n": len(rows),
            "accuracy_before": before,
            "accuracy_after": after,
            "accuracy_loss": before - after if rows else None,
            "passed": bool(rows) and before - after <= maximum_loss,
        }
    report_rows = patch_results["reports"]
    report_before = _mean([row["correct_before"] for row in report_rows])
    report_after = _mean([row["correct_after"] for row in report_rows])
    report_loss = report_before - report_after
    identity = controls_by_name["identity"]
    identity_pass = bool(identity) and not any(row["choice_changed"] for row in identity)
    identity_max_abs_margin_change = max(
        (abs(float(row["margin_change"])) for row in identity), default=float("inf")
    )
    identity_pass = identity_pass and identity_max_abs_margin_change <= 1e-4
    controls_pass = (
        all(item["passed"] for item in non_damage.values())
        and report_loss <= maximum_loss
        and identity_pass
    )
    reverse = controls_by_name["reverse_prose_into_success"]
    opposite = controls_by_name["opposite_target_same_base"]
    same_condition = controls_by_name["same_condition_cross_base"]
    return {
        "schema_version": 1,
        "study_id": "V5-RBG-5",
        "primary": {
            "n": len(primary),
            "n_base_clusters": len(cluster_values),
            "repair_rate": repair_rate,
            "mean_correct_action_margin_change": margin_change,
            "two_sided_exact_cluster_sign_flip_p": exact_p,
            "cluster_values": cluster_values,
            "normalized_recovery_n": len(normalized),
            "mean_normalized_recovery": _mean(normalized) if normalized else None,
            "passed": primary_pass,
        },
        "controls": {
            "identity_n": len(identity),
            "identity_max_absolute_margin_change": identity_max_abs_margin_change,
            "identity_passed": identity_pass,
            "non_damage": non_damage,
            "consequence_report_n": len(report_rows),
            "consequence_report_accuracy_before": report_before,
            "consequence_report_accuracy_after": report_after,
            "consequence_report_accuracy_loss": report_loss,
            "reverse_mean_margin_change": _mean(
                [row["margin_change"] for row in reverse]
            ),
            "opposite_target_repair_rate": _mean([row["repaired"] for row in opposite]),
            "same_condition_cross_base_repair_rate": _mean(
                [row["repaired"] for row in same_condition]
            )
            if same_condition
            else None,
            "passed": controls_pass,
        },
        "selective_causal_transport_passed": primary_pass and controls_pass,
        "claim_boundary": config["claim_boundary"],
    }
