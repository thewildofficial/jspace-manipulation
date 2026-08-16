import pytest

torch = pytest.importorskip("torch")

from jspace_policy.interventions import (  # noqa: E402
    MultiLayerResidualIntervention,
    ablate,
    coordinate_swap,
    j_lens_vector,
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


def test_multi_layer_intervention_updates_selected_positions_and_cleans_up() -> None:
    blocks = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
    hidden = torch.zeros(1, 3, 2)

    with MultiLayerResidualIntervention(
        blocks,
        {0: lambda value: value + 1, 1: lambda value: value + 2},
        positions=(-1,),
    ):
        for block in blocks:
            hidden = block(hidden)

    assert torch.equal(hidden[0, :2], torch.zeros(2, 2))
    assert torch.equal(hidden[0, -1], torch.full((2,), 3.0))
    assert all(not block._forward_hooks for block in blocks)


def test_multi_layer_intervention_can_transform_heterogeneous_batch() -> None:
    block = torch.nn.Identity()
    hidden = torch.zeros(2, 2, 1)

    def transform(value: torch.Tensor) -> torch.Tensor:
        value[0] += 1
        value[1, -1] += 2
        return value

    with MultiLayerResidualIntervention([block], {0: transform}):
        hidden = block(hidden)

    assert torch.equal(hidden[0], torch.ones(2, 1))
    assert torch.equal(hidden[1, 0], torch.zeros(1))
    assert torch.equal(hidden[1, 1], torch.full((1,), 2.0))
