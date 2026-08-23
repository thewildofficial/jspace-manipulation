from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from itertools import permutations, product
from typing import Any

SIGNALS = ("A", "B", "C")
RESPONSES = ("1", "2", "3")
FRAMES = ("strategic", "nonagentic")
SPLITS = ("discovery", "validation", "locked")
PAIR_TYPES = (
    "receiver_pathway",
    "payoff_pathway",
    "receiver_action_change",
    "payoff_action_change",
    "compression",
)
REPORT_QUERY_TYPES = (
    "decisive_response",
    "predicted_response",
    "decision_margin",
)
REPORT_ACCESS_CONDITIONS = (
    "retrospective",
    "answer_only",
    "matched_trajectory",
    "reconstruction",
)

_PAIR_CACHE: dict[str, tuple[MatchedPair, ...]] = {}


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _stable_id(*parts: object) -> str:
    return hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:14]


def _stable_permutation(*parts: object) -> tuple[int, int, int]:
    digest = hashlib.sha256(":".join(map(str, parts)).encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    values = [0, 1, 2]
    rng.shuffle(values)
    return tuple(values)  # type: ignore[return-value]


def _display_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def report_spec(
    row: dict[str, Any],
    query_type: str,
    access_condition: str,
    selected_action: str,
    *,
    decision_prompt: str | None = None,
    trajectory: str | None = None,
    matched_trajectory: str | None = None,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic forced-choice report and reconstruction control."""
    if query_type not in REPORT_QUERY_TYPES:
        raise ValueError(f"unknown report query: {query_type}")
    if access_condition not in REPORT_ACCESS_CONDITIONS:
        raise ValueError(f"unknown report access condition: {access_condition}")
    if selected_action not in SIGNALS:
        raise ValueError(f"invalid selected action: {selected_action}")

    selected_index = SIGNALS.index(selected_action)
    policy = tuple(tuple(Fraction(value) for value in values) for values in row["policy"])
    values = tuple(Fraction(value) for value in row["values"])
    ranked = sorted(range(3), key=lambda index: (-values[index], index))
    comparison_index = next(index for index in ranked if index != selected_index)
    contributions = [
        (policy[selected_index][index] - policy[comparison_index][index])
        * int(row["payoffs"][index])
        for index in range(3)
    ]
    decisive_response = max(
        range(3), key=lambda index: (abs(contributions[index]), -index)
    )
    predicted_response = max(
        range(3), key=lambda index: (policy[selected_index][index], -index)
    )
    decision_margin = values[selected_index] - values[comparison_index]

    permutation = _stable_permutation(row["condition_id"], query_type)
    if query_type == "decisive_response":
        targets: tuple[Any, Any, Any] = (0, 1, 2)
        correct_value: Any = decisive_response
        rendered_values = tuple(f"R{index + 1}" for index in targets)
        question = (
            "Looking back at the choice, which response made the largest absolute "
            "contribution to the expected-payoff difference between the selected "
            "option and its strongest alternative?"
        )
    elif query_type == "predicted_response":
        targets = (0, 1, 2)
        correct_value = predicted_response
        rendered_values = tuple(f"R{index + 1}" for index in targets)
        subject = "receiver" if row["frame"] == "strategic" else "device"
        question = (
            f"Under the selected option, which {subject} response/outcome had the "
            "highest predicted probability?"
        )
    else:
        targets = (decision_margin - 1, decision_margin, decision_margin + 1)
        correct_value = decision_margin
        rendered_values = tuple(_display_fraction(value) for value in targets)
        question = (
            "What was the expected-payoff difference between the selected option "
            "and its strongest alternative?"
        )

    ordered_targets = tuple(targets[index] for index in permutation)
    ordered_rendered = tuple(rendered_values[index] for index in permutation)
    correct_index = ordered_targets.index(correct_value)
    expected_label = SIGNALS[correct_index]
    options = "; ".join(
        f"{label}={value}"
        for label, value in zip(SIGNALS, ordered_rendered, strict=True)
    )
    correct_surface = ordered_rendered[correct_index]
    report_question = (
        f"{question}\nOptions: {options}. Return only A, B, or C.\nAnswer:"
    )
    task = decision_prompt or row["prompt"]
    prefix = (
        [{"role": "system", "content": system_prompt}] if system_prompt else []
    )
    if access_condition == "retrospective":
        if trajectory is None:
            trajectory = selected_action
        messages = [
            *prefix,
            {"role": "user", "content": task},
            {"role": "assistant", "content": trajectory},
            {"role": "user", "content": report_question},
        ]
    elif access_condition == "answer_only":
        messages = [
            *prefix,
            {"role": "user", "content": task},
            {"role": "assistant", "content": f"FINAL: {selected_action}"},
            {"role": "user", "content": report_question},
        ]
    elif access_condition == "matched_trajectory":
        if matched_trajectory is None:
            raise ValueError("matched trajectory control requires assistant content")
        messages = [
            *prefix,
            {"role": "user", "content": task},
            {"role": "assistant", "content": matched_trajectory},
            {"role": "user", "content": report_question},
        ]
    else:
        reconstruction = (
            f"{task}\n"
            f"A prior decision-maker selected option {selected_action}. Independently "
            "reconstruct the requested quantity from the problem statement; you do "
            "not have access to that decision-maker's hidden computation.\n"
            f"{report_question}"
        )
        messages = [*prefix, {"role": "user", "content": reconstruction}]
    return {
        "report_id": _stable_id(
            row["condition_id"], query_type, access_condition, selected_action
        ),
        "condition_id": row["condition_id"],
        "query_type": query_type,
        "access_condition": access_condition,
        "selected_action": selected_action,
        "expected_label": expected_label,
        "correct_value": str(correct_value),
        "correct_surface": correct_surface,
        "options": {
            label: value for label, value in zip(SIGNALS, ordered_rendered, strict=True)
        },
        "messages": messages,
    }


@dataclass(frozen=True)
class GameState:
    policy: tuple[tuple[Fraction, Fraction, Fraction], ...]
    payoffs: tuple[int, int, int]
    costs: tuple[int, int, int]

    @property
    def values(self) -> tuple[Fraction, Fraction, Fraction]:
        return tuple(
            sum(
                probability * payoff
                for probability, payoff in zip(row, self.payoffs, strict=True)
            )
            - self.costs[index]
            for index, row in enumerate(self.policy)
        )  # type: ignore[return-value]

    @property
    def ranking(self) -> tuple[int, int, int]:
        return tuple(
            sorted(range(3), key=lambda index: (-self.values[index], index))
        )  # type: ignore[return-value]

    @property
    def winner(self) -> int:
        return self.ranking[0]

    @property
    def runner_up(self) -> int:
        return self.ranking[1]

    @property
    def margin(self) -> Fraction:
        return self.values[self.winner] - self.values[self.runner_up]

    @property
    def decisive_response(self) -> int:
        winner_row = self.policy[self.winner]
        runner_row = self.policy[self.runner_up]
        contributions = [
            (winner_row[index] - runner_row[index]) * self.payoffs[index]
            for index in range(3)
        ]
        return max(range(3), key=lambda index: (abs(contributions[index]), -index))

    def serializable(self) -> dict[str, Any]:
        return {
            "policy": [[str(value) for value in row] for row in self.policy],
            "payoffs": list(self.payoffs),
            "costs": list(self.costs),
            "values": [str(value) for value in self.values],
            "ranking": list(self.ranking),
            "winner": self.winner,
            "runner_up": self.runner_up,
            "margin": str(self.margin),
            "decisive_response": self.decisive_response,
        }


@dataclass(frozen=True)
class MatchedPair:
    pair_id: str
    pair_type: str
    split: str
    left: GameState
    right: GameState


@dataclass(frozen=True)
class SearchRow:
    schema_version: int
    condition_id: str
    matched_group_id: str
    pair_id: str
    pair_type: str
    side: str
    split: str
    frame: str
    wording: int
    prompt: str
    expected_action: str
    winner: int
    runner_up: int
    decisive_response: int
    margin: str
    values: tuple[str, str, str]
    policy: tuple[tuple[str, str, str], ...]
    payoffs: tuple[int, int, int]
    costs: tuple[int, int, int]
    marker_text: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["values"] = list(self.values)
        value["policy"] = [list(row) for row in self.policy]
        return value


def _matrix_pool(grid: Sequence[Fraction]) -> list[tuple[tuple[Fraction, ...], ...]]:
    rows = list(permutations(grid))
    return [tuple(matrix) for matrix in product(rows, repeat=3)]


def _valid(state: GameState, minimum_margin: Fraction) -> bool:
    return state.margin >= minimum_margin and len(set(state.values)) == 3


def _sample_payoffs(rng: random.Random, low: int, high: int) -> tuple[int, int, int]:
    return tuple(rng.sample(range(low, high + 1), 3))  # type: ignore[return-value]


def _sample_costs(rng: random.Random, grid: Sequence[int]) -> tuple[int, int, int]:
    return tuple(rng.choice(grid) for _ in range(3))  # type: ignore[return-value]


def _receiver_pair(
    rng: random.Random,
    matrices: Sequence[tuple[tuple[Fraction, ...], ...]],
    settings: dict[str, Any],
    *,
    same_action: bool,
) -> tuple[GameState, GameState]:
    minimum = Fraction(settings["minimum_margin"])
    for _ in range(10_000):
        payoffs = _sample_payoffs(rng, settings["payoff_low"], settings["payoff_high"])
        costs = _sample_costs(rng, settings["signal_cost_grid"])
        candidates = [
            GameState(matrix, payoffs, costs)
            for matrix in rng.sample(list(matrices), 80)
        ]
        candidates = [state for state in candidates if _valid(state, minimum)]
        rng.shuffle(candidates)
        for left in candidates:
            for right in candidates:
                if left.policy == right.policy:
                    continue
                if same_action:
                    if (left.winner, left.runner_up) != (right.winner, right.runner_up):
                        continue
                    if left.decisive_response == right.decisive_response:
                        continue
                elif left.winner == right.winner:
                    continue
                return left, right
    raise RuntimeError("could not find receiver pair")


def _payoff_pair(
    rng: random.Random,
    matrices: Sequence[tuple[tuple[Fraction, ...], ...]],
    settings: dict[str, Any],
    *,
    same_action: bool,
) -> tuple[GameState, GameState]:
    minimum = Fraction(settings["minimum_margin"])
    payoff_pool = list(
        permutations(range(settings["payoff_low"], settings["payoff_high"] + 1), 3)
    )
    for _ in range(10_000):
        matrix = rng.choice(matrices)
        costs = _sample_costs(rng, settings["signal_cost_grid"])
        candidates = [
            GameState(matrix, payoff, costs) for payoff in rng.sample(payoff_pool, 120)
        ]
        candidates = [state for state in candidates if _valid(state, minimum)]
        rng.shuffle(candidates)
        for left in candidates:
            for right in candidates:
                if left.payoffs == right.payoffs:
                    continue
                if same_action:
                    if (left.winner, left.runner_up) != (right.winner, right.runner_up):
                        continue
                    if left.decisive_response == right.decisive_response:
                        continue
                elif left.winner == right.winner:
                    continue
                return left, right
    raise RuntimeError("could not find payoff pair")


def _compression_pair(
    rng: random.Random,
    matrices: Sequence[tuple[tuple[Fraction, ...], ...]],
    settings: dict[str, Any],
) -> tuple[GameState, GameState]:
    minimum = Fraction(settings["minimum_margin"])
    tolerance = Fraction(settings["compression_margin_tolerance"])
    for _ in range(15_000):
        payoffs = _sample_payoffs(rng, settings["payoff_low"], settings["payoff_high"])
        costs = _sample_costs(rng, settings["signal_cost_grid"])
        candidates = [
            GameState(matrix, payoffs, costs)
            for matrix in rng.sample(list(matrices), 120)
        ]
        candidates = [state for state in candidates if _valid(state, minimum)]
        rng.shuffle(candidates)
        for left in candidates:
            for right in candidates:
                if left.policy == right.policy or left.winner != right.winner:
                    continue
                if abs(left.margin - right.margin) > tolerance:
                    continue
                if left.decisive_response == right.decisive_response:
                    continue
                return left, right
    raise RuntimeError("could not find compression pair")


def generate_pairs(config: dict[str, Any]) -> list[MatchedPair]:
    settings = config["dataset"]
    cache_key = canonical_sha256(settings)
    if cache_key in _PAIR_CACHE:
        return list(_PAIR_CACHE[cache_key])
    rng = random.Random(int(settings["seed"]))
    matrices = _matrix_pool(tuple(Fraction(value) for value in settings["probability_grid"]))
    count = int(settings["pairs_per_type_per_split"])
    pairs: list[MatchedPair] = []
    factories = {
        "receiver_pathway": lambda: _receiver_pair(rng, matrices, settings, same_action=True),
        "payoff_pathway": lambda: _payoff_pair(rng, matrices, settings, same_action=True),
        "receiver_action_change": lambda: _receiver_pair(
            rng, matrices, settings, same_action=False
        ),
        "payoff_action_change": lambda: _payoff_pair(
            rng, matrices, settings, same_action=False
        ),
        "compression": lambda: _compression_pair(rng, matrices, settings),
    }
    for split in settings["splits"]:
        for pair_type in settings["pair_types"]:
            for index in range(count):
                left, right = factories[pair_type]()
                pair_id = _stable_id(settings["seed"], split, pair_type, index)
                pairs.append(MatchedPair(pair_id, pair_type, split, left, right))
    _PAIR_CACHE[cache_key] = tuple(pairs)
    return pairs


def _probability(value: Fraction) -> str:
    return f"{float(value):.2f}"


def _render(state: GameState, frame: str, wording: int) -> tuple[str, dict[str, str]]:
    table = "; ".join(
        f"{SIGNALS[index]}=({_probability(row[0])},{_probability(row[1])},{_probability(row[2])})"
        for index, row in enumerate(state.policy)
    )
    payoff = f"R1={state.payoffs[0]}, R2={state.payoffs[1]}, R3={state.payoffs[2]}"
    costs = f"A={state.costs[0]}, B={state.costs[1]}, C={state.costs[2]}"
    policy_end = "POLICY_END"
    payoff_end = "PAYOFF_END"
    if frame == "strategic":
        introductions = (
            "You are a sender choosing a signal for a receiver.",
            "Choose what to communicate to another decision-maker.",
            "A listener will react probabilistically to one of your three messages.",
        )
        policy_label = "Receiver response probabilities P(R1,R2,R3 | signal)"
        payoff_label = "Your payoff from the receiver's response"
    elif frame == "nonagentic":
        introductions = (
            "You are a controller selecting an input for a stochastic device.",
            "Choose which command to send to an automated mechanism.",
            "A random device will emit one of three outcomes after your input.",
        )
        policy_label = "Device outcome probabilities P(R1,R2,R3 | input)"
        payoff_label = "Score assigned to each device outcome"
    else:
        raise ValueError(f"unknown frame: {frame}")
    prompt = (
        f"{introductions[wording]}\n"
        f"{policy_label}: {table}. {policy_end}\n"
        f"{payoff_label}: {payoff}. {payoff_end}\n"
        f"Signal/input costs: {costs}.\n"
        "For each option, compute expected payoff as the probability-weighted response "
        "payoff minus its cost. Return only the best option: A, B, or C.\nAnswer:"
    )
    return prompt, {
        "policy_end": policy_end,
        "payoff_end": payoff_end,
        "final_prompt": "Answer:",
    }


def dataset_payload(config: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for pair in generate_pairs(config):
        for side, state in (("left", pair.left), ("right", pair.right)):
            for frame in config["dataset"]["frames"]:
                wording = SPLITS.index(pair.split)
                prompt, markers = _render(state, frame, wording)
                row = SearchRow(
                    schema_version=1,
                    condition_id=_stable_id(pair.pair_id, side, frame),
                    matched_group_id=pair.pair_id,
                    pair_id=pair.pair_id,
                    pair_type=pair.pair_type,
                    side=side,
                    split=pair.split,
                    frame=frame,
                    wording=wording,
                    prompt=prompt,
                    expected_action=SIGNALS[state.winner],
                    winner=state.winner,
                    runner_up=state.runner_up,
                    decisive_response=state.decisive_response,
                    margin=str(state.margin),
                    values=tuple(str(value) for value in state.values),
                    policy=tuple(tuple(str(value) for value in row) for row in state.policy),
                    payoffs=state.payoffs,
                    costs=state.costs,
                    marker_text=markers,
                )
                rows.append(row.as_dict())
    payload = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "status": "generated_before_behavior",
        "config_sha256": canonical_sha256(config),
        "rows": sorted(rows, key=lambda row: row["condition_id"]),
    }
    payload["content_sha256"] = canonical_sha256(payload)
    return payload


def _pair_rows(
    rows: Iterable[dict[str, Any]], pair_id: str, frame: str
) -> list[dict[str, Any]]:
    return sorted(
        [row for row in rows if row["pair_id"] == pair_id and row["frame"] == frame],
        key=lambda row: row["side"],
    )


def verify_dataset_payload(payload: dict[str, Any], config: dict[str, Any]) -> None:
    claimed = payload.get("content_sha256")
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    if claimed != canonical_sha256(body):
        raise ValueError("dataset content hash mismatch")
    rows = payload["rows"]
    expected = (
        len(config["dataset"]["splits"])
        * len(config["dataset"]["pair_types"])
        * int(config["dataset"]["pairs_per_type_per_split"])
        * 2
        * len(config["dataset"]["frames"])
    )
    if len(rows) != expected:
        raise ValueError(f"expected {expected} rows, found {len(rows)}")
    if len({row["condition_id"] for row in rows}) != len(rows):
        raise ValueError("condition IDs are not unique")
    for pair_id in sorted({row["pair_id"] for row in rows}):
        strategic = _pair_rows(rows, pair_id, "strategic")
        nonagentic = _pair_rows(rows, pair_id, "nonagentic")
        if len(strategic) != 2 or len(nonagentic) != 2:
            raise ValueError(f"pair {pair_id} lacks a complete frame/side factorial")
        for srow, nrow in zip(strategic, nonagentic, strict=True):
            fields = (
                "side",
                "expected_action",
                "winner",
                "runner_up",
                "decisive_response",
                "margin",
                "values",
                "policy",
                "payoffs",
                "costs",
            )
            if any(srow[field] != nrow[field] for field in fields):
                raise ValueError(f"frame mismatch for pair {pair_id}")
        left, right = strategic
        pair_type = left["pair_type"]
        if pair_type in {"receiver_pathway", "payoff_pathway"}:
            if (left["winner"], left["runner_up"]) != (right["winner"], right["runner_up"]):
                raise ValueError(f"same-action certificate failed for {pair_id}")
            if left["decisive_response"] == right["decisive_response"]:
                raise ValueError(f"decisive-response certificate failed for {pair_id}")
        elif pair_type == "compression":
            if left["winner"] != right["winner"]:
                raise ValueError(f"compression action certificate failed for {pair_id}")
            if left["decisive_response"] == right["decisive_response"]:
                raise ValueError(f"compression pathway certificate failed for {pair_id}")
        else:
            if left["winner"] == right["winner"]:
                raise ValueError(f"action-change certificate failed for {pair_id}")
        if pair_type.startswith("receiver"):
            if left["payoffs"] != right["payoffs"] or left["costs"] != right["costs"]:
                raise ValueError(f"receiver isolation failed for {pair_id}")
        if pair_type.startswith("payoff"):
            if left["policy"] != right["policy"] or left["costs"] != right["costs"]:
                raise ValueError(f"payoff isolation failed for {pair_id}")
