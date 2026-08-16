"""Frozen Stage 1 analysis helpers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np


def percentile_interval(values: list[float], confidence: float = 0.95) -> tuple[float, float]:
    alpha = (1 - confidence) / 2
    low, high = np.quantile(np.asarray(values, dtype=float), [alpha, 1 - alpha])
    return float(low), float(high)


def scenario_bootstrap(
    rows: list[dict[str, Any]],
    statistic: Callable[[list[dict[str, Any]]], float],
    *,
    draws: int,
    seed: int,
) -> tuple[float, float, float]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["base_scenario_id"])].append(row)
    scenarios = sorted(grouped)
    if not scenarios:
        raise ValueError("bootstrap requires at least one base scenario")
    point = statistic(rows)
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(draws):
        selected = rng.integers(0, len(scenarios), size=len(scenarios))
        sample = [item for index in selected for item in grouped[scenarios[int(index)]]]
        samples.append(statistic(sample))
    low, high = percentile_interval(samples)
    return float(point), low, high


def paired_policy_effect(rows: list[dict[str, Any]], column: str) -> float:
    paired: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for row in rows:
        key = (str(row["base_scenario_id"]), int(row["world_state_id"]))
        paired[key][str(row["policy_id"])] = float(row[column])
    if not paired or any(set(value) != {"T", "M"} for value in paired.values()):
        raise ValueError("paired policy effect requires complete T/M pairs")
    return float(np.mean([value["M"] - value["T"] for value in paired.values()]))


def macro_ovr_auc(targets: list[int], probabilities: list[list[float]]) -> float:
    y = np.asarray(targets, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    aucs: list[float] = []
    for candidate in range(p.shape[1]):
        positive = y == candidate
        n_positive = int(positive.sum())
        n_negative = int((~positive).sum())
        if not n_positive or not n_negative:
            continue
        order = np.argsort(p[:, candidate], kind="mergesort")
        ranks = np.empty(len(order), dtype=float)
        sorted_scores = p[order, candidate]
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and sorted_scores[end] == sorted_scores[start]:
                end += 1
            ranks[order[start:end]] = (start + 1 + end) / 2
            start = end
        rank_sum = ranks[positive].sum()
        aucs.append(
            float((rank_sum - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative))
        )
    return float(np.mean(aucs))


def expected_calibration_error(
    targets: list[int], probabilities: list[list[float]], bins: int = 10
) -> float:
    y = np.asarray(targets, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    predictions = p.argmax(axis=1)
    confidence = p.max(axis=1)
    correct = predictions == y
    result = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for index in range(bins):
        if index == bins - 1:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if mask.any():
            result += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return result


def classification_metrics(
    rows: list[dict[str, Any]],
    *,
    target_column: str,
    prediction_column: str,
    probability_column: str,
) -> dict[str, float]:
    targets = [int(row[target_column]) for row in rows]
    predictions = [int(row[prediction_column]) for row in rows]
    probabilities = [row[probability_column] for row in rows]
    probability_array = np.asarray(probabilities, dtype=float)
    clipped = np.clip(probability_array[np.arange(len(rows)), targets], 1e-12, 1)
    return {
        "accuracy": float(np.mean(np.asarray(targets) == np.asarray(predictions))),
        "macro_ovr_auroc": macro_ovr_auc(targets, probabilities),
        "negative_log_likelihood": float(-np.log(clipped).mean()),
        "expected_calibration_error": expected_calibration_error(targets, probabilities),
    }
