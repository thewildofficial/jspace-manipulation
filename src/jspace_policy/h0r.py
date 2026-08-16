from __future__ import annotations

import hashlib
from typing import Any

ARGUMENT_CONTROL_DEFINITIONS = [
    {
        "category": "unseen_countries",
        "arguments": ["Japan", "Brazil", "Germany", "Kenya"],
        "functions": [
            {
                "name": "capital",
                "template": "The capital of {arg} is the city of",
                "answers": {
                    "Japan": "Tokyo",
                    "Brazil": "Brasilia",
                    "Germany": "Berlin",
                    "Kenya": "Nairobi",
                },
            },
            {
                "name": "language",
                "template": "Most people in {arg} speak",
                "answers": {
                    "Japan": "Japanese",
                    "Brazil": "Portuguese",
                    "Germany": "German",
                    "Kenya": "Swahili",
                },
            },
            {
                "name": "continent",
                "template": "{arg} is a country on the continent of",
                "answers": {
                    "Japan": "Asia",
                    "Brazil": "South",
                    "Germany": "Europe",
                    "Kenya": "Africa",
                },
            },
            {
                "name": "currency",
                "template": "The currency used in {arg} is the",
                "answers": {
                    "Japan": "Yen",
                    "Brazil": "Real",
                    "Germany": "Euro",
                    "Kenya": "Shilling",
                },
            },
        ],
    },
    {
        "category": "greek_sequence",
        "arguments": ["alpha", "beta", "gamma", "delta"],
        "functions": [
            {
                "name": "successor",
                "template": "The Greek letter immediately after {arg} is",
                "answers": {
                    "alpha": "beta",
                    "beta": "gamma",
                    "gamma": "delta",
                    "delta": "epsilon",
                },
            },
            {
                "name": "ordinal",
                "template": "Counting Greek letters from alpha, {arg} is number",
                "answers": {
                    "alpha": "one",
                    "beta": "two",
                    "gamma": "three",
                    "delta": "four",
                },
            },
        ],
    },
    {
        "category": "chemical_elements",
        "arguments": ["hydrogen", "carbon", "oxygen", "sodium"],
        "functions": [
            {
                "name": "symbol",
                "template": "The chemical symbol for {arg} is",
                "answers": {
                    "hydrogen": "H",
                    "carbon": "C",
                    "oxygen": "O",
                    "sodium": "Na",
                },
            },
            {
                "name": "atomic_number",
                "template": "The atomic number of {arg} is",
                "answers": {
                    "hydrogen": "one",
                    "carbon": "six",
                    "oxygen": "eight",
                    "sodium": "eleven",
                },
            },
        ],
    },
    {
        "category": "weekdays",
        "arguments": ["Monday", "Wednesday", "Friday", "Sunday"],
        "functions": [
            {
                "name": "successor",
                "template": "The day immediately after {arg} is",
                "answers": {
                    "Monday": "Tuesday",
                    "Wednesday": "Thursday",
                    "Friday": "Saturday",
                    "Sunday": "Monday",
                },
            },
            {
                "name": "ordinal",
                "template": "Counting the week from Monday, {arg} is day number",
                "answers": {
                    "Monday": "one",
                    "Wednesday": "three",
                    "Friday": "five",
                    "Sunday": "seven",
                },
            },
        ],
    },
]


PARITY_CANDIDATES = [
    (3, 5),
    (2, 5),
    (7, 9),
    (4, 9),
    (11, 13),
    (10, 13),
    (6, 18),
    (6, 17),
    (14, 22),
    (14, 23),
    (21, 27),
    (21, 28),
    (32, 44),
    (32, 45),
    (51, 63),
    (51, 64),
    (8, 26),
    (8, 27),
    (17, 35),
    (17, 36),
    (24, 48),
    (24, 49),
    (39, 57),
    (39, 58),
    (42, 68),
    (42, 69),
    (55, 77),
    (55, 78),
    (64, 86),
    (64, 87),
    (71, 93),
    (71, 94),
    (12, 30),
    (12, 31),
    (25, 43),
    (25, 44),
    (36, 72),
    (36, 73),
    (47, 65),
    (47, 66),
]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _one_token(tokenizer: Any, value: str) -> tuple[int, str]:
    for surface in (f" {value}", value):
        ids = tokenizer.encode(surface, add_special_tokens=False)
        if len(ids) == 1:
            return int(ids[0]), surface
    raise ValueError(f"no accepted one-token form: {value}")


def _continuation_token(tokenizer: Any, prompt: str, value: str) -> tuple[int, str]:
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    for surface in (f" {value}", value):
        full = tokenizer.encode(prompt + surface, add_special_tokens=False)
        if full[: len(prefix)] == prefix and len(full) == len(prefix) + 1:
            return int(full[-1]), surface
    raise ValueError(f"not a one-token continuation: {value}")


def _subsequence_positions(sequence: list[int], subsequence: list[int]) -> list[int]:
    hits = []
    for start in range(len(sequence) - len(subsequence) + 1):
        if sequence[start : start + len(subsequence)] == subsequence:
            hits.extend(range(start, start + len(subsequence)))
    return hits


def build_locked_controls(tokenizer: Any) -> dict[str, object]:
    argument_trials: list[dict[str, object]] = []
    argument_exclusions: list[dict[str, str]] = []
    for category in ARGUMENT_CONTROL_DEFINITIONS:
        arguments = category["arguments"]
        for function in category["functions"]:
            for source in arguments:
                prompt = function["template"].format(arg=source)
                prompt_ids = [int(token) for token in tokenizer.encode(prompt)]
                source_piece = tokenizer.encode(source, add_special_tokens=False)
                source_positions = _subsequence_positions(prompt_ids, source_piece)
                if not source_positions:
                    source_piece = tokenizer.encode(f" {source}", add_special_tokens=False)
                    source_positions = _subsequence_positions(prompt_ids, source_piece)
                for target in arguments:
                    if target == source:
                        continue
                    try:
                        source_token, source_surface = _one_token(tokenizer, source)
                        target_token, target_surface = _one_token(tokenizer, target)
                        source_answer_token, source_answer_surface = _continuation_token(
                            tokenizer, prompt, function["answers"][source]
                        )
                        target_answer_token, target_answer_surface = _continuation_token(
                            tokenizer, prompt, function["answers"][target]
                        )
                        if source_answer_token == target_answer_token:
                            raise ValueError("source and target answers share a token")
                        if not source_positions:
                            raise ValueError("argument token span not found")
                    except ValueError as exc:
                        argument_exclusions.append(
                            {
                                "category": str(category["category"]),
                                "function": str(function["name"]),
                                "source": source,
                                "target": target,
                                "reason": str(exc),
                            }
                        )
                        continue
                    key = f"{category['category']}:{function['name']}:{source}:{target}"
                    argument_trials.append(
                        {
                            "trial_id": _sha256(key)[:16],
                            "scenario_id": _sha256(
                                f"{category['category']}:{function['name']}:{source}"
                            )[:16],
                            "category": category["category"],
                            "function": function["name"],
                            "prompt": prompt,
                            "prompt_sha256": _sha256(prompt),
                            "prompt_token_ids": prompt_ids,
                            "argument_positions": source_positions,
                            "source_argument": source,
                            "source_argument_token_id": source_token,
                            "source_argument_surface": source_surface,
                            "target_argument": target,
                            "target_argument_token_id": target_token,
                            "target_argument_surface": target_surface,
                            "source_answer": function["answers"][source],
                            "source_answer_token_id": source_answer_token,
                            "source_answer_surface": source_answer_surface,
                            "target_answer": function["answers"][target],
                            "target_answer_token_id": target_answer_token,
                            "target_answer_surface": target_answer_surface,
                        }
                    )
    if len(argument_trials) < 100:
        raise RuntimeError(
            f"only {len(argument_trials)} argument trials survived; require >=100"
        )

    intermediate_trials: list[dict[str, object]] = []
    intermediate_exclusions: list[dict[str, str]] = []
    for a, b in PARITY_CANDIDATES:
        source_intermediate = "even" if (a + b) % 2 == 0 else "odd"
        target_intermediate = "odd" if source_intermediate == "even" else "even"
        source_answer = "RED" if source_intermediate == "even" else "BLUE"
        target_answer = "BLUE" if source_answer == "RED" else "RED"
        prompt = (
            "Rule: if the sum is even output RED; if the sum is odd output BLUE. "
            f"Compute silently. {a} + {b} =>"
        )
        prompt_ids = [int(token) for token in tokenizer.encode(prompt)]
        try:
            source_token, source_surface = _one_token(tokenizer, source_intermediate)
            target_token, target_surface = _one_token(tokenizer, target_intermediate)
            source_answer_token, source_answer_surface = _continuation_token(
                tokenizer, prompt, source_answer
            )
            target_answer_token, target_answer_surface = _continuation_token(
                tokenizer, prompt, target_answer
            )
            if source_answer_token == target_answer_token:
                raise ValueError("natural and counterfactual answers share a token")
        except ValueError as exc:
            intermediate_exclusions.append(
                {"a": str(a), "b": str(b), "reason": str(exc)}
            )
            continue
        key = f"parity:{a}:{b}:{source_intermediate}:{target_intermediate}"
        intermediate_trials.append(
            {
                "trial_id": _sha256(key)[:16],
                "scenario_id": _sha256(f"parity:{a}:{b}")[:16],
                "family": "sum_parity_intermediate",
                "prompt": prompt,
                "prompt_sha256": _sha256(prompt),
                "prompt_token_ids": prompt_ids,
                "a": a,
                "b": b,
                "sum": a + b,
                "source_intermediate": source_intermediate,
                "source_intermediate_token_id": source_token,
                "source_intermediate_surface": source_surface,
                "target_intermediate": target_intermediate,
                "target_intermediate_token_id": target_token,
                "target_intermediate_surface": target_surface,
                "source_answer": source_answer,
                "source_answer_token_id": source_answer_token,
                "source_answer_surface": source_answer_surface,
                "target_answer": target_answer,
                "target_answer_token_id": target_answer_token,
                "target_answer_surface": target_answer_surface,
            }
        )
    if len(intermediate_trials) < 20:
        raise RuntimeError(
            f"only {len(intermediate_trials)} intermediate trials survived; require >=20"
        )
    return {
        "schema_version": 1,
        "status": "locked_unopened",
        "argument_control": {
            "minimum_baseline_accuracy": 0.80,
            "trials": argument_trials,
            "tokenization_exclusions": argument_exclusions,
        },
        "intermediate_control": {
            "family": "sum_parity_intermediate",
            "trials": intermediate_trials,
            "tokenization_exclusions": intermediate_exclusions,
        },
        "source_definitions": {
            "argument_control": ARGUMENT_CONTROL_DEFINITIONS,
            "parity_candidates": PARITY_CANDIDATES,
        },
    }
