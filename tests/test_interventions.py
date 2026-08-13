import pytest

torch = pytest.importorskip("torch")

from jspace_policy.interventions import (  # noqa: E402
    ablate,
    coordinate_swap,
    family_j_lens_vector,
    j_lens_vector,
    norm_matched_random,
)


def test_j_lens_vector_is_composed_row() -> None:
    jacobian = torch.eye(3)
    unembed = torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]])
    vector = j_lens_vector(jacobian, unembed, 1, unit_norm=False)
    assert torch.equal(vector, unembed[1])


def test_ablation_removes_coordinate() -> None:
    hidden = torch.tensor([[2.0, 3.0]])
    direction = torch.tensor([1.0, 0.0])
    result = ablate(hidden, direction)
    assert torch.allclose(result, torch.tensor([[0.0, 3.0]]))


def test_coordinate_swap_preserves_orthogonal_component() -> None:
    hidden = torch.tensor([[2.0, 5.0, 7.0]])
    source = torch.tensor([1.0, 0.0, 0.0])
    target = torch.tensor([0.0, 1.0, 0.0])
    result = coordinate_swap(hidden, source, target)
    assert torch.allclose(result, torch.tensor([[5.0, 2.0, 7.0]]))


def test_family_vector_is_difference_of_mean_token_vectors() -> None:
    jacobian = torch.eye(3)
    unembed = torch.tensor(
        [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
    )
    result = family_j_lens_vector(
        jacobian, unembed, [0, 1], [2], unit_norm=False
    )
    assert torch.allclose(result, torch.tensor([2.0, -2.0, 0.0]))


def test_norm_matched_random_is_reproducible_and_matched() -> None:
    reference = torch.tensor([3.0, 4.0, 0.0])
    first = norm_matched_random(reference, seed=7)
    second = norm_matched_random(reference, seed=7)
    assert torch.equal(first, second)
    assert first.norm() == pytest.approx(reference.norm())
