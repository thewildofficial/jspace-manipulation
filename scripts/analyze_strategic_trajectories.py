"""Integrity checks for the V2-E2 Strategic J-Lens Trajectories MVP."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SELECTED_LAYERS = {"34", "42", "46", "54", "60"}
RELATIONAL_TERMS = {
    "r1",
    "r2",
    "r3",
    "response",
    "responses",
    "responder",
    "receiver",
    "probability",
    "probabilities",
    "payoff",
    "value",
    "expected",
    "likely",
}
RESPONSE_IDENTIFIERS = {"r1", "r2", "r3"}
AGENT_NOUNS = {"receiver", "responder", "player", "opponent"}


def _load(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def _normalized(token: str) -> str:
    return token.strip().casefold()


def _integrity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures = []
    if len(rows) != 48:
        failures.append(f"expected 48 rows, found {len(rows)}")
    if len({row["condition_id"] for row in rows}) != len(rows):
        failures.append("condition IDs are not unique")
    if any(len(row["all_layer_final_prompt"]) != 63 for row in rows):
        failures.append("not every row has 63 final-prompt layer readouts")
    if any(
        set(trace["layers"]) != SELECTED_LAYERS
        for row in rows
        for trace in row["trace"]
    ):
        failures.append("a trajectory contains a non-frozen selected-layer set")
    if any(row["synchronization"]["pre_final_trace_index"] is None for row in rows):
        failures.append("a rollout has no contextual pre-FINAL synchronization point")
    return {"pass": not failures, "failures": failures}


def _selected_concept_occurrences(
    rows: list[dict[str, Any]], *, mode: str | None = None
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        if mode is not None and row["reasoning_mode"] != mode:
            continue
        for trace in row["trace"]:
            for tokens in trace["layers"].values():
                for item in tokens:
                    token = _normalized(item["token"])
                    if token in RELATIONAL_TERMS:
                        counts[token] += 1
    return counts


def _agent_noun_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    output = {}
    for framing in ("strategic", "non_strategic"):
        counts: Counter[str] = Counter()
        for row in rows:
            if row["framing"] != framing:
                continue
            for trace in row["trace"]:
                for tokens in trace["layers"].values():
                    for item in tokens:
                        token = _normalized(item["token"])
                        if token in AGENT_NOUNS:
                            counts[token] += 1
        output[framing] = dict(sorted(counts.items()))
    return output


def _legal_action_summary(
    rows: list[dict[str, Any]], mode: str, point: str
) -> dict[str, float | int]:
    selected = [row for row in rows if row["reasoning_mode"] == mode]
    correct = 0
    probabilities = []
    for row in selected:
        trace_index = (
            0
            if point == "final_prompt"
            else int(row["synchronization"]["pre_final_trace_index"])
        )
        trace = row["trace"][trace_index]
        logits = trace["legal_action_logits"]
        choice = max(logits, key=logits.get)
        correct += choice == row["expected_label"]
        probabilities.append(
            math.exp(trace["legal_action_log_probs"][row["expected_label"]])
        )
    return {
        "n": len(selected),
        "legal_argmax_correct": correct,
        "mean_expected_legal_probability": sum(probabilities) / len(probabilities),
        "minimum_expected_legal_probability": min(probabilities),
    }


def _expected_label_visibility(
    rows: list[dict[str, Any]], mode: str, point: str
) -> dict[str, int]:
    selected = [row for row in rows if row["reasoning_mode"] == mode]
    any_layer = 0
    layer_60 = 0
    for row in selected:
        trace_index = (
            0
            if point == "final_prompt"
            else int(row["synchronization"]["pre_final_trace_index"])
        )
        layer_hits = {
            layer: any(
                item["token"].strip() == row["expected_label"] for item in tokens
            )
            for layer, tokens in row["trace"][trace_index]["layers"].items()
        }
        any_layer += any(layer_hits.values())
        layer_60 += layer_hits["60"]
    return {"n": len(selected), "any_selected_layer": any_layer, "layer_60": layer_60}


def _short_cot_presurface_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    matches: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row["reasoning_mode"] != "short_cot":
            continue
        prefix = ""
        for trace in row["trace"]:
            if trace["surface_token"]:
                prefix += trace["surface_token"]
            for tokens in trace["layers"].values():
                for item in tokens:
                    token = _normalized(item["token"])
                    if token in RELATIONAL_TERMS and token not in prefix.casefold():
                        matches[token].add(row["condition_id"])
    return {term: len(condition_ids) for term, condition_ids in sorted(matches.items())}


def _same_action_certification(dataset_rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric = {}
    for row in dataset_rows:
        numeric.setdefault(row["instance_id"], row)
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in numeric.values():
        if row["pair_kind"] != "action_change":
            by_pair[row["pair_id"]].append(row)
    certified = 0
    details = []
    for pair_id, pair in sorted(by_pair.items()):
        if len(pair) != 2:
            details.append(
                {
                    "pair_id": pair_id,
                    "valid": False,
                    "failure": f"expected two numerical instances, found {len(pair)}",
                }
            )
            continue
        left, right = sorted(pair, key=lambda row: row["pair_member"])
        shared_checks = {
            "winner_signal_same": left["winner_signal"] == right["winner_signal"],
            "runner_up_signal_same": (
                left["runner_up_signal"] == right["runner_up_signal"]
            ),
            "expected_label_same": left["expected_label"] == right["expected_label"],
            "runner_up_label_same": (
                left["runner_up_label"] == right["runner_up_label"]
            ),
            "signal_to_label_same": (
                left["signal_to_label"] == right["signal_to_label"]
            ),
            "decisive_response_different": (
                left["decisive_response"] != right["decisive_response"]
            ),
            "decisive_fraction_at_least_half_both": all(
                float(row["decisive_fraction"]) >= 0.50 for row in pair
            ),
            "margin_at_least_three_both": all(
                float(row["margin"]) >= 3.0 for row in pair
            ),
            "costs_same": left["costs"] == right["costs"],
        }
        if left["pair_kind"] == "receiver_causal":
            causal_checks = {
                "payoffs_same": left["payoffs"] == right["payoffs"],
                "probability_matrix_different": (
                    left["probabilities"] != right["probabilities"]
                ),
            }
        elif left["pair_kind"] == "payoff_causal":
            causal_checks = {
                "probability_matrix_same": (
                    left["probabilities"] == right["probabilities"]
                ),
                "payoffs_different": left["payoffs"] != right["payoffs"],
            }
        else:
            causal_checks = {"recognized_pair_kind": False}
        checks = {**shared_checks, **causal_checks}
        valid = all(checks.values())
        certified += valid
        details.append(
            {
                "pair_id": pair_id,
                "pair_kind": left["pair_kind"],
                "valid": valid,
                "checks": checks,
                "expected_label": left["expected_label"],
                "decisive_responses": [
                    left["decisive_response"],
                    right["decisive_response"],
                ],
            }
        )
    return {
        "certified_pairs": certified,
        "expected_pairs": 4,
        "all_pairs_valid": certified == 4 and len(details) == 4,
        "details": details,
    }


def analyze(mechanistic: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
    rows = mechanistic["rows"]
    direct_concepts = _selected_concept_occurrences(rows, mode="direct")
    all_concepts = _selected_concept_occurrences(rows)
    return {
        "schema_version": 1,
        "study_id": "V2-E2-MVP",
        "analysis_class": "exploratory_open_inspection_no_formal_onset_statistic",
        "mechanistic_run_id": mechanistic["metadata"]["run_id"],
        "dataset_sha256": mechanistic["metadata"]["dataset_sha256"],
        "integrity": _integrity(rows),
        "same_action_certification": _same_action_certification(dataset["rows"]),
        "selected_top20": {
            "direct_relational_occurrences": dict(sorted(direct_concepts.items())),
            "single_token_exact_response_identifier_occurrences": sum(
                all_concepts[term] for term in RESPONSE_IDENTIFIERS
            ),
            "single_token_exact_response_identifier_diagnostic_only": True,
            "single_token_exact_response_identifier_limitation": (
                "R1/R2/R3 may be split across tokenizer tokens; this count cannot "
                "support representational absence."
            ),
            "agent_noun_occurrences": _agent_noun_counts(rows),
            "rows_with_exact_concept_before_first_surface_occurrence": (
                _short_cot_presurface_rows(rows)
            ),
            "pre_surface_occurrence_limitation": (
                "Exact lexical comparison only; no minimum lead time is required and "
                "the statistic does not establish anticipation of reasoning."
            ),
        },
        "legal_action_readout": {
            mode: {
                point: _legal_action_summary(rows, mode, point)
                for point in ("final_prompt", "pre_final")
            }
            for mode in ("direct", "short_cot")
        },
        "expected_label_top20_visibility": {
            mode: {
                point: _expected_label_visibility(rows, mode, point)
                for point in ("final_prompt", "pre_final")
            }
            for mode in ("direct", "short_cot")
        },
        "final_classification": (
            "generic_task_semantics_and_action_preparation_without_decisive-pathway readout"
        ),
        "interpretation_boundary": (
            "This classifies only the stored top-k J-lens readouts; it does not establish "
            "absence from the residual stream or causal non-use."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mechanistic",
        default="results/v2_strategic_trajectories/raw/mechanistic.json.gz",
    )
    parser.add_argument(
        "--dataset", default="configs/v2/strategic_trajectories/dataset.json"
    )
    parser.add_argument(
        "--output",
        default="results/v2_strategic_trajectories/summaries/exploratory_summary.json",
    )
    args = parser.parse_args()
    summary = analyze(_load(Path(args.mechanistic)), _load(Path(args.dataset)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
