import pytest

torch = pytest.importorskip("torch")

from jspace_policy.h0r_diagnostics import (  # noqa: E402
    FP32_SWAP_ATOL,
    basis_matrix,
    describe_basis,
    donor_coordinate_patch,
    reconstruction_fraction,
    resolve_position_mask,
    symmetric_coordinate_swap,
)


def test_swap_exchanges_nonorthogonal_coordinates_before_cast() -> None:
    source = torch.tensor([1.0, 0.0, 0.0])
    target = torch.tensor([0.5, 1.0, 0.0])
    matrix = basis_matrix(source, target)
    hidden = torch.tensor([[2.0, 5.0, 7.0]], dtype=torch.float32)
    expected_pre = hidden @ torch.linalg.pinv(matrix).T

    updated, diagnostics = symmetric_coordinate_swap(hidden, source, target)
    actual_post = updated.float() @ torch.linalg.pinv(matrix).T

    assert torch.allclose(actual_post, expected_pre.flip(-1), atol=FP32_SWAP_ATOL)
    assert diagnostics["coordinates"]["fp32_exchange_max_abs_error"] <= FP32_SWAP_ATOL
    assert diagnostics["coordinates"]["source_post"] == pytest.approx(
        diagnostics["coordinates"]["target_pre"], abs=FP32_SWAP_ATOL
    )


def test_interpolated_swap_achieves_requested_coordinate_target() -> None:
    source = torch.tensor([1.0, 0.0])
    target = torch.tensor([0.0, 1.0])
    hidden = torch.tensor([[2.0, 6.0]])
    updated, _ = symmetric_coordinate_swap(hidden, source, target, alpha=0.25)
    assert torch.allclose(updated, torch.tensor([[3.0, 5.0]]))


def test_conditioning_is_reported_and_rank_deficiency_rejected() -> None:
    diagnostics = describe_basis(
        torch.tensor([1.0, 0.0]), torch.tensor([0.99, 0.01])
    )
    assert diagnostics.absolute_cosine > 0.99
    assert diagnostics.condition_number > 100
    with pytest.raises(ValueError, match="rank deficient"):
        basis_matrix(torch.tensor([1.0, 0.0]), torch.tensor([2.0, 0.0]))


def test_donor_patch_matches_donor_coordinates_only() -> None:
    source = torch.tensor([1.0, 0.0, 0.0])
    target = torch.tensor([0.0, 1.0, 0.0])
    hidden = torch.tensor([[2.0, 5.0, 11.0]])
    donor = torch.tensor([[7.0, 3.0, -4.0]])
    updated, diagnostics = donor_coordinate_patch(hidden, donor, source, target)
    assert torch.allclose(updated, torch.tensor([[7.0, 3.0, 11.0]]))
    assert diagnostics["donor_source_coordinate"] == [7.0]


@pytest.mark.parametrize(
    ("mask", "expected"),
    [
        ("final_prompt_position", [5]),
        ("argument_token_position", [2, 3]),
        ("argument_through_end", [2, 3, 4, 5]),
        ("all_prompt_positions", [0, 1, 2, 3, 4, 5]),
    ],
)
def test_position_masks(mask: str, expected: list[int]) -> None:
    assert resolve_position_mask(
        mask, sequence_length=6, argument_positions=[2, 3]
    ) == expected


def test_reconstruction_fraction_and_exclusion() -> None:
    assert reconstruction_fraction(0.0, 2.0, -2.0) == pytest.approx(0.5)
    assert reconstruction_fraction(1.0, 1.0, 1.0) is None
