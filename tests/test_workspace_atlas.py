from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from jspace_policy.workspace_atlas import (
    ACTIONS,
    GAMES,
    SPLITS,
    dataset_payload,
    verify_dataset_payload,
)


def _config() -> dict[str, object]:
    return {
        "games": list(GAMES),
        "rows_per_game_per_split": 12,
        "dataset_seed": 61031,
    }


def test_dataset_is_deterministic_and_balanced() -> None:
    first = dataset_payload(_config())
    second = dataset_payload(_config())
    assert first == second
    assert len(first["rows"]) == 6 * 3 * 12
    for game in GAMES:
        for split in SPLITS:
            assert (
                sum(row["game"] == game and row["split"] == split for row in first["rows"])
                == 12
            )


def test_solver_answers_are_self_consistent() -> None:
    payload = dataset_payload(_config())
    verify_dataset_payload(payload)
    assert all(len(row["solver"]["action_values"]) == 3 for row in payload["rows"])
    assert all(row["expected_action"] in {"A", "B", "C"} for row in payload["rows"])


def _independent_best(values: list[float]) -> str:
    return ACTIONS[max(range(3), key=lambda index: (values[index], -index))]


def test_every_solver_payoff_is_independently_recomputable() -> None:
    rows = dataset_payload(_config())["rows"]
    rank = {"J": 0, "Q": 1, "K": 2}
    for row in rows:
        solver = row["solver"]
        stored = [float(value) for value in solver["action_values"]]
        if row["game"] == "inspection":
            p = float(solver["audit_probability"])
            # The prompt and solver must encode the same arithmetic expression.
            assert f"{p:.2f}" in row["prompt"]
        elif row["game"] == "kuhn":
            card = row["private_state"]
            opponents = solver["opponent_cards"]
            calls = solver["call_table"]
            check = sum(1.0 if rank[card] > rank[o] else -1.0 for o in opponents) / 2
            bet = (
                sum(
                    (1 - calls[o]) + calls[o] * (2.0 if rank[card] > rank[o] else -2.0)
                    for o in opponents
                )
                / 2
            )
            assert stored == [bet, check, -2.0]
        elif row["game"] == "cheap_talk":
            preferred = row["objective"]
            actions = solver["receiver_actions"]
            assert stored == [-float((action - preferred) ** 2) for action in actions]
        elif row["game"] == "signaling":
            costs = solver["costs"]
            impressions = solver["impressions"]
            assert stored == [
                2.0 * impression - costs[i] for i, impression in enumerate(impressions)
            ]
        elif row["game"] == "disclosure":
            approvals = solver["receiver_approve"]
            costs = solver["disclosure_costs"]
            recomputed = [
                (5.0 if approved else 0.0)
                - (costs[0] if i == 0 else costs[1] if i == 1 else sum(costs))
                for i, approved in enumerate(approvals)
            ]
            assert all(math.isclose(a, b) for a, b in zip(stored, recomputed, strict=True))
        elif row["game"] == "chameleon":
            expected_values = (
                [0.35, 0.75, 0.40] if solver["role"] == "insider" else [0.70, 0.45, 0.10]
            )
            assert stored == expected_values
        else:  # pragma: no cover - exhaustive guard
            raise AssertionError(row["game"])
        assert row["expected_action"] == _independent_best(stored)
        ordered = sorted(stored, reverse=True)
        assert math.isclose(row["value_margin"], ordered[0] - ordered[1])


def test_parameter_inventory_is_identical_across_rendering_splits() -> None:
    rows = dataset_payload(_config())["rows"]
    inventories: dict[tuple[str, str], Counter[str]] = {}
    for game in GAMES:
        for split in SPLITS:
            inventories[(game, split)] = Counter(
                json.dumps(row["solver"], sort_keys=True)
                for row in rows
                if row["game"] == game and row["split"] == split
            )
        assert inventories[(game, "discovery")] == inventories[(game, "validation")]
        assert inventories[(game, "discovery")] == inventories[(game, "locked_replication")]


def test_core_games_have_behavioral_and_strategic_dissociations() -> None:
    rows = [row for row in dataset_payload(_config())["rows"] if row["split"] == "discovery"]
    for game in GAMES:
        game_rows = [row for row in rows if row["game"] == game]
        assert len({row["expected_action"] for row in game_rows}) >= 2

    # At least one action has multiple strategic interpretations.
    by_game_action: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        by_game_action[(row["game"], row["expected_action"])].add(row["strategy"])
    assert any(len(strategies) > 1 for strategies in by_game_action.values())

    # At least one private state changes action as beliefs/incentives change.
    by_game_state: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        by_game_state[(row["game"], row["private_state"])].add(row["expected_action"])
    assert any(len(actions) > 1 for actions in by_game_state.values())


def test_prompts_are_complete_and_do_not_embed_an_answer_key() -> None:
    rows = dataset_payload(_config())["rows"]
    for row in rows:
        assert row["prompt"].endswith("Answer:")
        assert "Return exactly one letter: A, B, or C" in row["prompt"]
        assert "correct answer" not in row["prompt"].lower()
        assert "expected action" not in row["prompt"].lower()
        assert all(math.isfinite(float(value)) for value in row["solver"]["action_values"])
        assert float(row["value_margin"]) > 0


def test_matched_groups_are_pairs_and_never_cross_splits() -> None:
    rows = dataset_payload(_config())["rows"]
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[row["matched_group_id"]].append(row)
    assert all(len(group) == 2 for group in groups.values())
    assert all(len({row["split"] for row in group}) == 1 for group in groups.values())
    assert all(len({row["game"] for row in group}) == 1 for group in groups.values())


def test_frozen_config_generates_valid_payload() -> None:
    config = json.loads(Path("configs/v2/workspace_atlas/experiment.json").read_text())
    verify_dataset_payload(dataset_payload(config))
