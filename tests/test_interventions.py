import pytest

torch = pytest.importorskip("torch")

from jspace_policy.interventions import ablate, coordinate_swap, j_lens_vector  # noqa: E402


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
