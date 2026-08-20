"""Frozen local protocol machinery for V2-E2 Strategic J-Space Trajectories.

The module deliberately contains no model or lens dependencies.  It defines the
exact numerical corpus, prompt renderings, parsers, and behavioral gate that must
be fixed and tested before any activation is inspected.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import cache
from itertools import permutations, product
from typing import Any

SIGNALS = ("S1", "S2", "S3")
RESPONSES = ("R1", "R2", "R3")
LABELS = ("A", "B", "C")
FRAMINGS = ("strategic", "non_strategic")
REASONING_MODES = ("direct", "short_cot")

BANNED_PROMPT_TERMS = (
    "optimal",
    "optimize",
    "maximize",
    "best",
    "highest expected",
    "recommended",
    "correct answer",
    "should choose",
    "winning option",
    "strategy",
)

DIRECT_RE = re.compile(r"^FINAL:\s*([ABC])$")
FINAL_LINE_RE = re.compile(r"(?m)^FINAL:\s*([ABC])\s*$")


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _stable_id(prefix: str, *parts: object) -> str:
    digest = hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:16]
    return f"{prefix}-{digest}"


def _fraction(value: str | int | Fraction) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(value)


def _decimal_text(value: Fraction) -> str:
    """Render the finite decimals used by the frozen probability grid exactly."""

    scaled = value * 100
    if scaled.denominator != 1:
        raise ValueError(f"probability is not an exact hundredth: {value}")
    sign = "-" if scaled.numerator < 0 else ""
    absolute = abs(scaled.numerator)
    return f"{sign}{absolute // 100}.{absolute % 100:02d}"


def _signed(value: int) -> str:
    return f"{value:+d}" if value != 0 else "0"


@dataclass(frozen=True)
class NumericInstance:
    instance_id: str
    pair_id: str
    pair_kind: str
    member: int
    payoff_state: str
    probabilities: tuple[tuple[Fraction, Fraction, Fraction], ...]
    payoffs: tuple[int, int, int]
    costs: tuple[int, int, int]
    signal_to_label: tuple[str, str, str]


@dataclass(frozen=True)
class DecisionCertificate:
    q_values: tuple[Fraction, Fraction, Fraction]
    winner: int
    runner_up: int
    margin: Fraction
    response_support: tuple[Fraction, Fraction, Fraction]
    cost_support: Fraction
    positive_support: Fraction
    decisive_response: int
    decisive_fraction: Fraction


def solve_instance(
    probabilities: Sequence[Sequence[Fraction | str | int]],
    payoffs: Sequence[int],
    costs: Sequence[int],
) -> DecisionCertificate:
    """Solve one three-signal decision and certify its decisive response pathway."""

    if len(probabilities) != 3 or any(len(row) != 3 for row in probabilities):
        raise ValueError("probabilities must be a 3x3 matrix")
    if len(payoffs) != 3 or len(costs) != 3:
        raise ValueError("payoffs and costs must each have length three")
    matrix = tuple(tuple(_fraction(value) for value in row) for row in probabilities)
    if any(sum(row) != 1 for row in matrix):
        raise ValueError("every probability row must sum exactly to one")
    if any(value < 0 or value > 1 for row in matrix for value in row):
        raise ValueError("probabilities must lie in [0, 1]")

    q_values = tuple(
        sum((matrix[s][r] * payoffs[r] for r in range(3)), Fraction()) - costs[s]
        for s in range(3)
    )
    order = sorted(range(3), key=lambda index: (-q_values[index], index))
    if q_values[order[0]] == q_values[order[1]]:
        raise ValueError("decision does not have a unique winner")
    winner, runner_up = order[:2]
    response_support = tuple(
        (matrix[winner][r] - matrix[runner_up][r]) * payoffs[r] for r in range(3)
    )
    cost_support = Fraction(-(costs[winner] - costs[runner_up]))
    positive_support = sum((max(value, 0) for value in response_support), Fraction()) + max(
        cost_support, 0
    )
    positive_responses = tuple(max(value, 0) for value in response_support)
    decisive_response = max(range(3), key=lambda index: (positive_responses[index], -index))
    if positive_support <= 0 or positive_responses[decisive_response] <= 0:
        raise ValueError("decision has no positive response pathway")
    if positive_responses.count(positive_responses[decisive_response]) != 1:
        raise ValueError("decision does not have a unique decisive response pathway")
    return DecisionCertificate(
        q_values=q_values,
        winner=winner,
        runner_up=runner_up,
        margin=q_values[winner] - q_values[runner_up],
        response_support=response_support,
        cost_support=cost_support,
        positive_support=positive_support,
        decisive_response=decisive_response,
        decisive_fraction=positive_responses[decisive_response] / positive_support,
    )


def _valid_certificate(certificate: DecisionCertificate, minimum_margin: Fraction) -> bool:
    return (
        certificate.margin >= minimum_margin
        and certificate.decisive_fraction >= Fraction(1, 2)
    )


def _fast_certificate(
    probabilities: Sequence[Sequence[Fraction]],
    payoffs: Sequence[int],
    costs: Sequence[int],
    minimum_margin: Fraction,
) -> tuple[int, int, int] | None:
    """Integer-scaled search predicate; accepted rows are rechecked by the exact solver."""

    scale = 100
    frozen_matrix = tuple(tuple(row) for row in probabilities)
    matrix = _scaled_probability_matrix(frozen_matrix)
    q_values = [
        sum(matrix[signal][response] * payoffs[response] for response in range(3))
        - scale * costs[signal]
        for signal in range(3)
    ]
    order = sorted(range(3), key=lambda index: (-q_values[index], index))
    winner, runner_up = order[:2]
    if q_values[winner] == q_values[runner_up]:
        return None
    if Fraction(q_values[winner] - q_values[runner_up], scale) < minimum_margin:
        return None
    support = [
        (matrix[winner][response] - matrix[runner_up][response]) * payoffs[response]
        for response in range(3)
    ]
    cost_support = -scale * (costs[winner] - costs[runner_up])
    positive = [max(value, 0) for value in support]
    total_positive = sum(positive) + max(cost_support, 0)
    maximum = max(positive)
    if total_positive <= 0 or maximum <= 0 or positive.count(maximum) != 1:
        return None
    if 2 * maximum < total_positive:
        return None
    return winner, runner_up, positive.index(maximum)


@cache
def _scaled_probability_matrix(
    probabilities: tuple[tuple[Fraction, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(value.numerator * 100 // value.denominator for value in row)
        for row in probabilities
    )


def _random_label_mapping(rng: random.Random) -> tuple[str, str, str]:
    labels = list(LABELS)
    rng.shuffle(labels)
    return tuple(labels)  # type: ignore[return-value]


def _random_payoff(rng: random.Random, low: int, high: int) -> tuple[int, int, int]:
    values = rng.sample(range(low, high + 1), 3)
    return tuple(values)  # type: ignore[return-value]


def _probability_matrices(grid: tuple[Fraction, Fraction, Fraction]) -> list[tuple[Any, ...]]:
    rows = list(permutations(grid))
    return [tuple(matrix) for matrix in product(rows, repeat=3)]


def _find_response_causal_pair(
    rng: random.Random,
    matrices: Sequence[tuple[Any, ...]],
    payoff_low: int,
    payoff_high: int,
    costs_grid: Sequence[int],
    minimum_margin: Fraction,
) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[int, int, int], tuple[int, int, int]]:
    for _ in range(5_000):
        payoffs = _random_payoff(rng, payoff_low, payoff_high)
        costs = tuple(rng.choice(costs_grid) for _ in SIGNALS)
        candidates: list[tuple[tuple[Any, ...], tuple[int, int, int]]] = []
        for matrix in matrices:
            certificate = _fast_certificate(matrix, payoffs, costs, minimum_margin)
            if certificate is not None:
                candidates.append((matrix, certificate))
        rng.shuffle(candidates)
        grouped: dict[tuple[int, int], dict[int, tuple[Any, ...]]] = {}
        for matrix, (winner, runner_up, decisive_response) in candidates:
            key = (winner, runner_up)
            grouped.setdefault(key, {}).setdefault(decisive_response, matrix)
        for pathways in grouped.values():
            if len(pathways) >= 2:
                left, right = list(pathways.values())[:2]
                return left, right, payoffs, costs  # type: ignore[return-value]
    raise RuntimeError("could not generate a receiver-causal matched pair")


def _find_payoff_causal_pair(
    rng: random.Random,
    matrices: Sequence[tuple[Any, ...]],
    payoff_low: int,
    payoff_high: int,
    costs_grid: Sequence[int],
    minimum_margin: Fraction,
) -> tuple[
    tuple[Any, ...], tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]
]:
    payoff_pool = list(permutations(range(payoff_low, payoff_high + 1), 3))
    for _ in range(5_000):
        matrix = rng.choice(matrices)
        costs = tuple(rng.choice(costs_grid) for _ in SIGNALS)
        rng.shuffle(payoff_pool)
        candidates: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
        for payoffs in payoff_pool[:1_000]:
            certificate = _fast_certificate(matrix, payoffs, costs, minimum_margin)
            if certificate is not None:
                candidates.append((payoffs, certificate))
        grouped: dict[tuple[int, int], dict[int, tuple[int, int, int]]] = {}
        for payoffs, (winner, runner_up, decisive_response) in candidates:
            key = (winner, runner_up)
            grouped.setdefault(key, {}).setdefault(decisive_response, payoffs)
        for pathways in grouped.values():
            if len(pathways) >= 2:
                left, right = list(pathways.values())[:2]
                return matrix, left, right, costs  # type: ignore[return-value]
    raise RuntimeError("could not generate a payoff-causal matched pair")


def _find_action_change_pair(
    rng: random.Random,
    matrices: Sequence[tuple[Any, ...]],
    payoff_low: int,
    payoff_high: int,
    costs_grid: Sequence[int],
    minimum_margin: Fraction,
) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[int, int, int], tuple[int, int, int]]:
    for _ in range(5_000):
        payoffs = _random_payoff(rng, payoff_low, payoff_high)
        costs = tuple(rng.choice(costs_grid) for _ in SIGNALS)
        candidates: list[tuple[tuple[Any, ...], tuple[int, int, int]]] = []
        for matrix in matrices:
            certificate = _fast_certificate(matrix, payoffs, costs, minimum_margin)
            if certificate is not None:
                candidates.append((matrix, certificate))
        rng.shuffle(candidates)
        by_winner: dict[int, tuple[Any, ...]] = {}
        for matrix, (winner, _runner_up, _decisive_response) in candidates:
            by_winner.setdefault(winner, matrix)
        if len(by_winner) >= 2:
            left, right = list(by_winner.values())[:2]
            return left, right, payoffs, costs  # type: ignore[return-value]
    raise RuntimeError("could not generate a different-action control pair")


def _payoff_state(payoffs: Sequence[int]) -> str:
    digest = hashlib.sha256(json.dumps(list(payoffs)).encode()).hexdigest()[:8].upper()
    return f"P{digest}"


def generate_instances(config: dict[str, Any]) -> list[NumericInstance]:
    """Search for the prospectively constrained six matched pairs."""

    settings = config["dataset"]
    rng = random.Random(int(settings["seed"]))
    grid = tuple(Fraction(value) for value in settings["probability_grid"])
    if len(grid) != 3 or sum(grid) != 1:
        raise ValueError("the probability grid must contain three values summing to one")
    matrices = _probability_matrices(grid)  # type: ignore[arg-type]
    rng.shuffle(matrices)
    payoff_low, payoff_high = map(int, settings["payoff_range"])
    costs_grid = tuple(map(int, settings["cost_grid"]))
    minimum_margin = Fraction(str(settings["minimum_margin"]))
    requested = settings["pair_counts"]
    instances: list[NumericInstance] = []

    def append_pair(
        kind: str,
        pair_number: int,
        left_matrix: tuple[Any, ...],
        right_matrix: tuple[Any, ...],
        left_payoff: tuple[int, int, int],
        right_payoff: tuple[int, int, int],
        costs: tuple[int, int, int],
    ) -> None:
        version = int(settings.get("version", 1))
        pair_id = f"v{version}-pair-{pair_number:02d}-{kind}"
        mapping = _random_label_mapping(rng)
        for member, matrix, payoffs in (
            (1, left_matrix, left_payoff),
            (2, right_matrix, right_payoff),
        ):
            instances.append(
                NumericInstance(
                    instance_id=f"v{version}-instance-{pair_number:02d}-{member}",
                    pair_id=pair_id,
                    pair_kind=kind,
                    member=member,
                    payoff_state=_payoff_state(payoffs),
                    probabilities=matrix,  # type: ignore[arg-type]
                    payoffs=payoffs,
                    costs=costs,
                    signal_to_label=mapping,
                )
            )

    pair_number = 0
    for _ in range(int(requested["receiver_causal"])):
        pair_number += 1
        left, right, payoffs, costs = _find_response_causal_pair(
            rng, matrices, payoff_low, payoff_high, costs_grid, minimum_margin
        )
        append_pair("receiver_causal", pair_number, left, right, payoffs, payoffs, costs)
    for _ in range(int(requested["payoff_causal"])):
        pair_number += 1
        matrix, left, right, costs = _find_payoff_causal_pair(
            rng, matrices, payoff_low, payoff_high, costs_grid, minimum_margin
        )
        append_pair("payoff_causal", pair_number, matrix, matrix, left, right, costs)
    for _ in range(int(requested["action_change"])):
        pair_number += 1
        left, right, payoffs, costs = _find_action_change_pair(
            rng, matrices, payoff_low, payoff_high, costs_grid, minimum_margin
        )
        append_pair("action_change", pair_number, left, right, payoffs, payoffs, costs)
    verify_instances(instances, config)
    return instances


def verify_instances(instances: Sequence[NumericInstance], config: dict[str, Any]) -> None:
    settings = config["dataset"]
    minimum_margin = Fraction(str(settings["minimum_margin"]))
    expected_pairs = sum(map(int, settings["pair_counts"].values()))
    if len(instances) != expected_pairs * 2:
        raise ValueError("unexpected number of numerical instances")
    by_pair: dict[str, list[NumericInstance]] = {}
    for instance in instances:
        by_pair.setdefault(instance.pair_id, []).append(instance)
    if len(by_pair) != expected_pairs or any(len(pair) != 2 for pair in by_pair.values()):
        raise ValueError("instances do not form complete matched pairs")

    winning_labels: list[str] = []
    for pair in by_pair.values():
        left, right = sorted(pair, key=lambda item: item.member)
        left_cert = solve_instance(left.probabilities, left.payoffs, left.costs)
        right_cert = solve_instance(right.probabilities, right.payoffs, right.costs)
        if not _valid_certificate(left_cert, minimum_margin) or not _valid_certificate(
            right_cert, minimum_margin
        ):
            raise ValueError("instance violates the frozen margin/pathway threshold")
        if left.signal_to_label != right.signal_to_label:
            raise ValueError("action-label permutation changed within a pair")
        if left.pair_kind in {"receiver_causal", "payoff_causal"}:
            if (left_cert.winner, left_cert.runner_up) != (
                right_cert.winner,
                right_cert.runner_up,
            ):
                raise ValueError("same-action pair changed winner or runner-up")
            if left_cert.decisive_response == right_cert.decisive_response:
                raise ValueError("same-action pair did not change decisive response pathway")
        if left.pair_kind == "receiver_causal":
            if left.payoffs != right.payoffs or left.costs != right.costs:
                raise ValueError("receiver-causal pair changed payoff or costs")
            if left.probabilities == right.probabilities:
                raise ValueError("receiver-causal pair did not change response policy")
        elif left.pair_kind == "payoff_causal":
            if left.probabilities != right.probabilities or left.costs != right.costs:
                raise ValueError("payoff-causal pair changed probabilities or costs")
            if left.payoffs == right.payoffs:
                raise ValueError("payoff-causal pair did not change payoff state")
        elif left.pair_kind == "action_change":
            if left.payoffs != right.payoffs or left.costs != right.costs:
                raise ValueError("action-change pair changed payoff or costs")
            if left_cert.winner == right_cert.winner:
                raise ValueError("action-change pair did not change the action")
        else:
            raise ValueError(f"unknown pair kind: {left.pair_kind}")
        winning_labels.extend(
            (left.signal_to_label[left_cert.winner], right.signal_to_label[right_cert.winner])
        )
    if set(winning_labels) != set(LABELS):
        raise ValueError("all three rendered actions are not represented")


def _label_order(instance: NumericInstance) -> list[tuple[str, int]]:
    return sorted(((label, index) for index, label in enumerate(instance.signal_to_label)))


def render_prompt(
    *,
    payoff_state: str,
    probabilities: Sequence[Sequence[Fraction]],
    payoffs: Sequence[int],
    costs: Sequence[int],
    signal_to_label: Sequence[str],
    framing: str,
    reasoning_mode: str,
) -> str:
    """Render primitive inputs only; the function cannot receive solver metadata."""

    if framing == "strategic":
        introduction = (
            "Responder is another player. For each signal, it selects R1, R2, or R3 "
            "with the probabilities below."
        )
    elif framing == "non_strategic":
        introduction = (
            "Responder is a stochastic mechanism. For each signal, it selects R1, R2, or R3 "
            "with the probabilities below."
        )
    else:
        raise ValueError(f"unknown framing: {framing}")

    label_rows = sorted((label, index) for index, label in enumerate(signal_to_label))
    probability_blocks = []
    for label, signal_index in label_rows:
        values = probabilities[signal_index]
        probability_blocks.append(
            f"Signal {label}:\n"
            + "\n".join(
                f"- {response}: {_decimal_text(_fraction(values[index]))}"
                for index, response in enumerate(RESPONSES)
            )
        )
    cost_lines = "\n".join(
        f"- {label}: {costs[signal_index]}" for label, signal_index in label_rows
    )
    payoff_lines = "\n".join(
        f"- {response}: {_signed(int(payoffs[index]))}"
        for index, response in enumerate(RESPONSES)
    )
    if reasoning_mode == "direct":
        instruction = (
            "Do not show calculations or provide an explanation. Your entire response must be "
            "exactly one line in the form `FINAL: <letter>`, using A, B, or C."
        )
    elif reasoning_mode == "short_cot":
        instruction = (
            "Briefly reason through the situation in at most three short sentences and no more "
            "than 30 words. Then write a new final line in the form `FINAL: <letter>`, using "
            "A, B, or C."
        )
    else:
        raise ValueError(f"unknown reasoning mode: {reasoning_mode}")
    rendered_probabilities = "\n\n".join(probability_blocks)
    prompt = (
        f"Private payoff state: {payoff_state}.\n\n"
        f"{introduction}\n\n"
        "Your payoff for each selected output is:\n"
        f"{payoff_lines}\n\n"
        "Conditional output probabilities:\n\n"
        f"{rendered_probabilities}\n\n"
        "Signal costs:\n"
        f"{cost_lines}\n\n"
        "Choose a signal according to these scoring rules.\n"
        f"{instruction}"
    )
    lint_prompt(prompt)
    return prompt


def lint_prompt(prompt: str) -> None:
    lowered = prompt.casefold()
    found = [term for term in BANNED_PROMPT_TERMS if term.casefold() in lowered]
    if found:
        raise ValueError(f"prompt contains banned terms: {found}")
    counts = {
        label: len(re.findall(rf"(?<![A-Za-z0-9]){label}(?![A-Za-z0-9])", prompt))
        for label in LABELS
    }
    if len(set(counts.values())) != 1:
        raise ValueError(f"action labels are not symmetrically rendered: {counts}")


def dataset_payload(config: dict[str, Any]) -> dict[str, Any]:
    instances = generate_instances(config)
    rows: list[dict[str, Any]] = []
    for instance in instances:
        certificate = solve_instance(instance.probabilities, instance.payoffs, instance.costs)
        expected_label = instance.signal_to_label[certificate.winner]
        runner_up_label = instance.signal_to_label[certificate.runner_up]
        decisive_response = RESPONSES[certificate.decisive_response]
        for framing, reasoning_mode in product(FRAMINGS, REASONING_MODES):
            prompt = render_prompt(
                payoff_state=instance.payoff_state,
                probabilities=instance.probabilities,
                payoffs=instance.payoffs,
                costs=instance.costs,
                signal_to_label=instance.signal_to_label,
                framing=framing,
                reasoning_mode=reasoning_mode,
            )
            condition_id = _stable_id(
                "e2t",
                instance.instance_id,
                framing,
                reasoning_mode,
                config["dataset"]["seed"],
                config["dataset"].get("version", 1),
            )
            rows.append(
                {
                    "schema_version": 1,
                    "condition_id": condition_id,
                    "instance_id": instance.instance_id,
                    "pair_id": instance.pair_id,
                    "pair_kind": instance.pair_kind,
                    "pair_member": instance.member,
                    "payoff_state": instance.payoff_state,
                    "framing": framing,
                    "reasoning_mode": reasoning_mode,
                    "probabilities": [
                        [_decimal_text(value) for value in row]
                        for row in instance.probabilities
                    ],
                    "payoffs": list(instance.payoffs),
                    "costs": list(instance.costs),
                    "signal_to_label": dict(
                        zip(SIGNALS, instance.signal_to_label, strict=True)
                    ),
                    "q_values": [_decimal_text(value) for value in certificate.q_values],
                    "winner_signal": SIGNALS[certificate.winner],
                    "runner_up_signal": SIGNALS[certificate.runner_up],
                    "expected_label": expected_label,
                    "runner_up_label": runner_up_label,
                    "margin": _decimal_text(certificate.margin),
                    "decisive_response": decisive_response,
                    "decisive_fraction": float(certificate.decisive_fraction),
                    "response_support": [
                        _decimal_text(value) for value in certificate.response_support
                    ],
                    "cost_support": _decimal_text(certificate.cost_support),
                    "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                }
            )
    rows.sort(key=lambda row: row["condition_id"])
    payload: dict[str, Any] = {
        "schema_version": 1,
        "study_id": "V2-E2-MVP",
        "status": "untokenized_source_frozen_before_behavior",
        "dataset_version": int(config["dataset"].get("version", 1)),
        "dataset_seed": int(config["dataset"]["seed"]),
        "rows": rows,
    }
    verify_dataset_payload(payload, config)
    payload["content_sha256"] = canonical_sha256(payload)
    return payload


def verify_dataset_payload(payload: dict[str, Any], config: dict[str, Any]) -> None:
    rows = payload["rows"]
    if len(rows) != 48:
        raise ValueError(f"expected 48 rollout rows, found {len(rows)}")
    if len({row["condition_id"] for row in rows}) != len(rows):
        raise ValueError("condition IDs are not unique")
    if Counter(row["framing"] for row in rows) != {"strategic": 24, "non_strategic": 24}:
        raise ValueError("framing balance failed")
    if Counter(row["reasoning_mode"] for row in rows) != {"direct": 24, "short_cot": 24}:
        raise ValueError("reasoning-mode balance failed")
    for row in rows:
        lint_prompt(row["prompt"])
        if any(term in row["prompt"] for term in (str(row["q_values"]), row["margin"])):
            raise ValueError("solver metadata leaked into prompt")
    for instance_id in {row["instance_id"] for row in rows}:
        instance_rows = [row for row in rows if row["instance_id"] == instance_id]
        if len(instance_rows) != 4:
            raise ValueError("each numerical instance must have four rollout renderings")
        primitive_keys = (
            "probabilities",
            "payoffs",
            "costs",
            "signal_to_label",
            "expected_label",
        )
        if any(
            len({json.dumps(row[key], sort_keys=True) for row in instance_rows}) != 1
            for key in primitive_keys
        ):
            raise ValueError("isomorphic renderings changed numerical inputs or answer")


def completed_sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return deterministic completed-sentence spans under the frozen rule."""

    spans: list[tuple[int, int]] = []
    start = 0
    index = 0
    while index < len(text):
        char = text[index]
        boundary = False
        end = index + 1
        if char == "\n":
            boundary = True
        elif char in ".?!":
            decimal = (
                char == "."
                and index > 0
                and index + 1 < len(text)
                and text[index - 1].isdigit()
                and text[index + 1].isdigit()
            )
            followed_by_boundary = index + 1 == len(text) or text[index + 1].isspace()
            boundary = not decimal and followed_by_boundary
        if boundary:
            segment = text[start:end].strip()
            if segment:
                spans.append((start, end))
            start = end
        index += 1
    return spans


def first_completed_reasoning_sentence(text: str) -> str | None:
    spans = completed_sentence_spans(text.strip())
    return text.strip()[spans[0][0] : spans[0][1]].strip() if spans else None


def parse_output(text: str, reasoning_mode: str) -> dict[str, Any]:
    stripped = text.strip()
    occurrences = re.findall(r"FINAL:", stripped)
    if reasoning_mode == "direct":
        match = DIRECT_RE.fullmatch(stripped)
        return {
            "parseable": bool(match and len(occurrences) == 1),
            "label": match.group(1) if match else None,
            "reasoning_sentence_count": 0,
            "first_reasoning_sentence": None,
        }
    if reasoning_mode != "short_cot":
        raise ValueError(f"unknown reasoning mode: {reasoning_mode}")
    matches = list(FINAL_LINE_RE.finditer(stripped))
    final_match = matches[-1] if matches else None
    final_is_last = bool(final_match and not stripped[final_match.end() :].strip())
    reasoning = stripped[: final_match.start()] if final_match else ""
    spans = completed_sentence_spans(reasoning)
    parseable = bool(
        len(occurrences) == 1
        and len(matches) == 1
        and final_is_last
        and 1 <= len(spans) <= 3
    )
    return {
        "parseable": parseable,
        "label": final_match.group(1) if final_match else None,
        "reasoning_sentence_count": len(spans),
        "first_reasoning_sentence": (
            reasoning[spans[0][0] : spans[0][1]].strip() if spans else None
        ),
    }


def continuation_token_ids(
    encode: Callable[[str], Sequence[int]], rendered_context: str
) -> dict[str, int]:
    """Validate A/B/C as distinct one-token suffixes after the exact FINAL: prefix."""

    if not rendered_context.endswith("FINAL:"):
        raise ValueError("tokenization context must end exactly in `FINAL:`")
    prefix_ids = list(encode(rendered_context))
    output: dict[str, int] = {}
    for label in LABELS:
        full_ids = list(encode(rendered_context + " " + label))
        if full_ids[: len(prefix_ids)] != prefix_ids or len(full_ids) != len(prefix_ids) + 1:
            raise ValueError(f"{label} is not one continuation token after the frozen prefix")
        output[label] = int(full_ids[-1])
    if len(set(output.values())) != len(LABELS):
        raise ValueError("legal action token IDs are not distinct")
    return output


def behavior_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    if len(rows) != 48:
        raise ValueError(f"behavior gate requires all 48 rows, found {len(rows)}")

    def outcome(part: list[dict[str, Any]]) -> dict[str, int | float]:
        correct = sum(bool(row["correct"]) for row in part)
        return {"n": len(part), "correct": correct, "accuracy": correct / len(part)}

    overall = outcome(rows)
    marginals = {
        "strategic": outcome([row for row in rows if row["framing"] == "strategic"]),
        "non_strategic": outcome(
            [row for row in rows if row["framing"] == "non_strategic"]
        ),
        "direct": outcome([row for row in rows if row["reasoning_mode"] == "direct"]),
        "short_cot": outcome(
            [row for row in rows if row["reasoning_mode"] == "short_cot"]
        ),
    }
    parseable = sum(bool(row["parseable"]) for row in rows)
    gate = bool(
        parseable == 48
        and overall["correct"] >= 44
        and all(value["correct"] >= 21 for value in marginals.values())
    )
    return {
        "n_rows": 48,
        "parseable": parseable,
        "overall": overall,
        "marginals": marginals,
        "gate_pass": gate,
        "mechanistic_inspection_authorized": gate,
    }


def serialize_instance(instance: NumericInstance) -> dict[str, Any]:
    value = asdict(instance)
    value["probabilities"] = [
        [_decimal_text(cell) for cell in row] for row in instance.probabilities
    ]
    return value
