from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from jspace_policy.jspace_interventions import (  # noqa: E402
    balanced_unrelated_target,
    cluster_bootstrap_mean,
    coordinate_swap_fp32,
    matched_random_swap_fp32,
    projected_coordinates,
    resolve_position_mask,
    select_first_feasible_corpus,
)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0, 2.0])
def test_coordinate_swap_matches_analytic_formula(alpha: float) -> None:
    hidden = torch.tensor([[2.0, 5.0, 11.0]])
    source = torch.tensor([1.0, 0.0, 0.0])
    target = torch.tensor([0.5, 1.0, 0.0])
    matrix = torch.stack([source, target], dim=1)
    pre = projected_coordinates(hidden, matrix)
    updated, diagnostics = coordinate_swap_fp32(hidden, source, target, alpha=alpha)
    post = projected_coordinates(updated, matrix)
    expected = pre + alpha * (pre.flip(-1) - pre)
    assert torch.allclose(post, expected, atol=2e-5, rtol=2e-5)
    assert diagnostics.coordinate_target_error <= 2e-5


def test_alpha_zero_and_identity_are_noops() -> None:
    hidden = torch.randn(3, 5)
    source = torch.randn(5)
    target = torch.randn(5)
    zero, _ = coordinate_swap_fp32(hidden, source, target, alpha=0.0)
    identity, diagnostics = coordinate_swap_fp32(hidden, source, source, alpha=2.0)
    assert torch.equal(zero, hidden)
    assert torch.equal(identity, hidden)
    assert not diagnostics.geometry_eligible
    assert diagnostics.condition_number is None


def test_alpha_half_equalizes_and_alpha_one_swaps() -> None:
    hidden = torch.tensor([[2.0, 6.0]])
    source = torch.tensor([1.0, 0.0])
    target = torch.tensor([0.0, 1.0])
    half, _ = coordinate_swap_fp32(hidden, source, target, alpha=0.5)
    full, _ = coordinate_swap_fp32(hidden, source, target, alpha=1.0)
    double, _ = coordinate_swap_fp32(hidden, source, target, alpha=2.0)
    assert torch.allclose(half, torch.tensor([[4.0, 4.0]]))
    assert torch.allclose(full, torch.tensor([[6.0, 2.0]]))
    assert torch.allclose(double, torch.tensor([[10.0, -2.0]]))


def test_orthogonal_component_is_preserved() -> None:
    hidden = torch.tensor([[2.0, 5.0, 11.0]])
    source = torch.tensor([1.0, 0.0, 0.0])
    target = torch.tensor([0.0, 1.0, 0.0])
    updated, _ = coordinate_swap_fp32(hidden, source, target, alpha=2.0)
    assert updated[0, 2] == hidden[0, 2]


def test_random_control_is_deterministic_orthogonal_and_delta_matched() -> None:
    hidden = torch.randn(4, 32)
    source = torch.randn(32)
    target = torch.randn(32)
    first, first_diagnostics = matched_random_swap_fp32(
        hidden, source, target, alpha=1.0, seed=1729
    )
    second, second_diagnostics = matched_random_swap_fp32(
        hidden, source, target, alpha=1.0, seed=1729
    )
    assert torch.equal(first, second)
    assert first_diagnostics == second_diagnostics
    assert first_diagnostics["delta_l2_match_max_abs_error"] <= 2e-5


def test_balanced_unrelated_cycles_targets() -> None:
    arguments = ["a", "b", "c", "d"]
    assert [
        balanced_unrelated_target(arguments, "a", target) for target in ["b", "c", "d"]
    ] == ["c", "d", "b"]


def test_position_masks_include_final_exclusion_diagnostic() -> None:
    assert resolve_position_mask(
        "all_prompt_positions", sequence_length=5, argument_positions=[1]
    ) == [0, 1, 2, 3, 4]
    assert resolve_position_mask(
        "all_except_final_position", sequence_length=5, argument_positions=[1]
    ) == [0, 1, 2, 3]


def _candidate_fixture() -> tuple[dict[str, object], list[dict[str, object]]]:
    arguments = ["a", "b", "c", "d", "e"]
    functions = [
        {
            "name": f"f{index}",
            "template": "{arg}",
            "answers": {argument: f"{argument}{index}" for argument in arguments},
        }
        for index in range(5)
    ]
    candidates: dict[str, object] = {
        "study_id": "test",
        "categories": [{"name": "category", "arguments": arguments, "functions": functions}],
    }
    rows: list[dict[str, object]] = []
    token_id = 0
    for function in functions:
        for argument in arguments:
            token_id += 1
            rows.append(
                {
                    "category": "category",
                    "function": function["name"],
                    "argument": argument,
                    "tokenization_eligible": True,
                    "clean_correct": True,
                    "answer_token_id": token_id,
                }
            )
    return candidates, rows


def test_dataset_selection_is_deterministic_and_first_feasible() -> None:
    candidates, rows = _candidate_fixture()
    first = select_first_feasible_corpus(candidates, rows)
    second = select_first_feasible_corpus(candidates, rows)
    assert first == second
    category = first["categories"][0]
    assert category["arguments"] == ["a", "b", "c", "d"]
    assert [function["name"] for function in category["functions"]] == [
        "f0",
        "f1",
        "f2",
        "f3",
    ]


def test_bootstrap_keeps_target_swaps_in_clusters() -> None:
    values = [1.0, 2.0, 3.0, 10.0, 11.0, 12.0]
    clusters = ["a", "a", "a", "b", "b", "b"]
    assert cluster_bootstrap_mean(values, clusters, draws=200, seed=1729) == (
        pytest.approx(2.0),
        pytest.approx(11.0),
    )
