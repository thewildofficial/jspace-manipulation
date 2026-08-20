from __future__ import annotations

import json
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import pytest

from jspace_policy.strategic_trajectories import (
    LABELS,
    behavior_summary,
    completed_sentence_spans,
    continuation_token_ids,
    dataset_payload,
    generate_instances,
    lint_prompt,
    parse_output,
    solve_instance,
)


def _config() -> dict[str, object]:
    return json.loads(
        Path("configs/v2/strategic_trajectories/experiment.json").read_text()
    )


def test_exact_solver_and_decisive_pathway() -> None:
    certificate = solve_instance(
        (
            (Fraction("0.80"), Fraction("0.15"), Fraction("0.05")),
            (Fraction("0.05"), Fraction("0.80"), Fraction("0.15")),
            (Fraction("0.15"), Fraction("0.05"), Fraction("0.80")),
        ),
        (8, 1, -4),
        (0, 1, 0),
    )
    assert certificate.winner == 0
    assert certificate.margin >= 2
    assert certificate.decisive_response == 0
    assert certificate.decisive_fraction >= Fraction(1, 2)


def test_generator_certifies_all_six_pairs() -> None:
    config = _config()
    instances = generate_instances(config)
    assert len(instances) == 12
    groups = defaultdict(list)
    for instance in instances:
        groups[instance.pair_id].append(instance)
    assert Counter(pair[0].pair_kind for pair in groups.values()) == {
        "receiver_causal": 2,
        "payoff_causal": 2,
        "action_change": 2,
    }
    winners = []
    for pair in groups.values():
        left, right = sorted(pair, key=lambda row: row.member)
        left_certificate = solve_instance(left.probabilities, left.payoffs, left.costs)
        right_certificate = solve_instance(right.probabilities, right.payoffs, right.costs)
        winners.extend(
            (
                left.signal_to_label[left_certificate.winner],
                right.signal_to_label[right_certificate.winner],
            )
        )
        if left.pair_kind != "action_change":
            assert left_certificate.winner == right_certificate.winner
            assert left_certificate.runner_up == right_certificate.runner_up
            assert left_certificate.decisive_response != right_certificate.decisive_response
        else:
            assert left_certificate.winner != right_certificate.winner
    assert set(winners) == set(LABELS)


def test_payload_is_balanced_isomorphic_and_deterministic() -> None:
    config = _config()
    left = dataset_payload(config)
    right = dataset_payload(config)
    assert left == right
    assert len(left["rows"]) == 48
    assert Counter(row["framing"] for row in left["rows"]) == {
        "strategic": 24,
        "non_strategic": 24,
    }
    assert Counter(row["reasoning_mode"] for row in left["rows"]) == {
        "direct": 24,
        "short_cot": 24,
    }
    by_instance = defaultdict(list)
    for row in left["rows"]:
        by_instance[row["instance_id"]].append(row)
    for rows in by_instance.values():
        assert len({json.dumps(row["probabilities"]) for row in rows}) == 1
        assert len({tuple(row["payoffs"]) for row in rows}) == 1
        assert len({row["expected_label"] for row in rows}) == 1


def test_checked_in_source_payload_matches_generator() -> None:
    frozen = json.loads(
        Path("configs/v2/strategic_trajectories/dataset_source.json").read_text()
    )
    assert frozen == dataset_payload(_config())


def test_prompts_are_symmetric_and_differ_only_in_framing_sentence() -> None:
    rows = dataset_payload(_config())["rows"]
    by_key = defaultdict(dict)
    for row in rows:
        lint_prompt(row["prompt"])
        by_key[(row["instance_id"], row["reasoning_mode"])][row["framing"]] = row[
            "prompt"
        ]
    for prompts in by_key.values():
        strategic = prompts["strategic"]
        control = prompts["non_strategic"]
        assert strategic.replace("another player", "a stochastic mechanism") == control


@pytest.mark.parametrize(
    ("text", "mode", "valid", "label"),
    (
        ("FINAL: A", "direct", True, "A"),
        ("Reason.\nFINAL: B", "direct", False, None),
        ("One sentence.\nFINAL: C", "short_cot", True, "C"),
        ("Value is 2.5. Then B.\nFINAL: B", "short_cot", True, "B"),
        ("No punctuation\nFINAL: A", "short_cot", True, "A"),
        ("One. Two. Three. Four.\nFINAL: A", "short_cot", False, "A"),
        ("One.\nFINAL: A\nextra", "short_cot", False, "A"),
        ("One.\nFINAL: A\nFINAL: B", "short_cot", False, "B"),
    ),
)
def test_frozen_output_parser(text: str, mode: str, valid: bool, label: str | None) -> None:
    parsed = parse_output(text, mode)
    assert parsed["parseable"] is valid
    assert parsed["label"] == label


def test_sentence_boundaries_ignore_decimal_points() -> None:
    assert len(completed_sentence_spans("R1 yields 2.5. Then compare.\n")) == 2


def test_contextual_continuation_validation() -> None:
    vocabulary = {"prefixFINAL:": [1, 2], " A": 10, " B": 11, " C": 12}

    def encode(text: str) -> list[int]:
        if text == "prefixFINAL:":
            return vocabulary[text]
        suffix = text[-2:]
        return [1, 2, vocabulary[suffix]]

    assert continuation_token_ids(encode, "prefixFINAL:") == {"A": 10, "B": 11, "C": 12}
    with pytest.raises(ValueError, match="end exactly"):
        continuation_token_ids(encode, "prefix")


def test_behavior_gate_is_conjunctive() -> None:
    payload = dataset_payload(_config())
    passing = [
        {
            "framing": row["framing"],
            "reasoning_mode": row["reasoning_mode"],
            "parseable": True,
            "correct": True,
        }
        for row in payload["rows"]
    ]
    assert behavior_summary(passing)["gate_pass"]
    passing[0]["parseable"] = False
    assert not behavior_summary(passing)["gate_pass"]
    passing[0]["parseable"] = True
    for row in passing[:5]:
        row["correct"] = False
    assert not behavior_summary(passing)["gate_pass"]
