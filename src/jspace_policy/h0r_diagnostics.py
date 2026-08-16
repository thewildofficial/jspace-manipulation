from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch

FP32_SWAP_ATOL = 2e-5
FP32_SWAP_RTOL = 2e-5


@dataclass(frozen=True)
class BasisDiagnostics:
    cosine: float
    absolute_cosine: float
    condition_number: float


@dataclass(frozen=True)
class CoordinateDiagnostics:
    source_pre: list[float]
    target_pre: list[float]
    source_post: list[float]
    target_post: list[float]
    fp32_exchange_max_abs_error: float
    hidden_l2: float
    delta_l2: float
    delta_rms_ratio: float


def basis_matrix(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return a full-rank fp32 two-vector basis on the vectors' device."""
    if source.ndim != 1 or target.ndim != 1 or source.shape != target.shape:
        raise ValueError("source and target must be same-width vectors")
    matrix = torch.stack([source.float(), target.float()], dim=1)
    if int(torch.linalg.matrix_rank(matrix).item()) != 2:
        raise ValueError("source/target basis is rank deficient")
    return matrix


def projected_coordinates(hidden: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Project residual states into the least-squares coordinates of ``matrix``."""
    if hidden.shape[-1] != matrix.shape[0] or matrix.shape[1] != 2:
        raise ValueError("hidden and two-vector basis have incompatible shapes")
    return hidden.float() @ torch.linalg.pinv(matrix).T


def describe_basis(source: torch.Tensor, target: torch.Tensor) -> BasisDiagnostics:
    matrix = basis_matrix(source, target)
    cosine = torch.nn.functional.cosine_similarity(
        matrix[:, 0], matrix[:, 1], dim=0
    )
    return BasisDiagnostics(
        cosine=float(cosine.detach().cpu()),
        absolute_cosine=float(cosine.abs().detach().cpu()),
        condition_number=float(torch.linalg.cond(matrix).detach().cpu()),
    )


def _coordinate_record(
    hidden: torch.Tensor,
    updated: torch.Tensor,
    matrix: torch.Tensor,
    expected_post: torch.Tensor,
) -> CoordinateDiagnostics:
    before = hidden.float()
    after = updated.float()
    pre = projected_coordinates(before, matrix)
    post = projected_coordinates(after, matrix)
    error = (post - expected_post).abs().max()
    delta = after - before
    before_rms = before.square().mean(dim=-1).sqrt().clamp_min(1e-12)
    ratio = delta.square().mean(dim=-1).sqrt() / before_rms
    return CoordinateDiagnostics(
        source_pre=pre[..., 0].detach().cpu().reshape(-1).tolist(),
        target_pre=pre[..., 1].detach().cpu().reshape(-1).tolist(),
        source_post=post[..., 0].detach().cpu().reshape(-1).tolist(),
        target_post=post[..., 1].detach().cpu().reshape(-1).tolist(),
        fp32_exchange_max_abs_error=float(error.detach().cpu()),
        hidden_l2=float(before.norm(dim=-1).mean().detach().cpu()),
        delta_l2=float(delta.norm(dim=-1).mean().detach().cpu()),
        delta_rms_ratio=float(ratio.mean().detach().cpu()),
    )


def symmetric_coordinate_swap(
    hidden: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float = 1.0,
    assert_exchange: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Swap local source/target coordinates in fp32 and return complete diagnostics.

    The assertion is evaluated before casting the result back to the model dtype.
    At alpha values other than one it checks the exact interpolated coordinate target.
    """
    matrix = basis_matrix(source, target).to(hidden.device)
    value = hidden.float()
    pre = projected_coordinates(value, matrix)
    expected_post = pre + float(alpha) * (pre.flip(-1) - pre)
    updated_fp32 = value + (expected_post - pre) @ matrix.T
    record = _coordinate_record(value, updated_fp32, matrix, expected_post)
    if assert_exchange and not torch.allclose(
        projected_coordinates(updated_fp32, matrix),
        expected_post,
        atol=FP32_SWAP_ATOL,
        rtol=FP32_SWAP_RTOL,
    ):
        raise AssertionError(
            "fp32 coordinate exchange failed: "
            f"max abs error={record.fp32_exchange_max_abs_error:.6g}"
        )
    return updated_fp32.to(hidden.dtype), {
        "basis": asdict(describe_basis(source, target)),
        "coordinates": asdict(record),
    }


def directional_coordinate_write(
    hidden: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    source_shift: float,
    target_shift: float,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Suppress source and install target coordinates by frozen absolute shifts."""
    matrix = basis_matrix(source, target).to(hidden.device)
    value = hidden.float()
    pre = projected_coordinates(value, matrix)
    shift = torch.tensor(
        [-float(source_shift), float(target_shift)], device=value.device
    )
    expected_post = pre + float(alpha) * shift
    updated_fp32 = value + (expected_post - pre) @ matrix.T
    record = _coordinate_record(value, updated_fp32, matrix, expected_post)
    return updated_fp32.to(hidden.dtype), {
        "basis": asdict(describe_basis(source, target)),
        "coordinates": asdict(record),
    }


def donor_coordinate_patch(
    hidden: torch.Tensor,
    donor_hidden: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Patch the two-coordinate state toward a paired natural donor state."""
    matrix = basis_matrix(source, target).to(hidden.device)
    value = hidden.float()
    pre = projected_coordinates(value, matrix)
    donor = projected_coordinates(donor_hidden.float().to(value.device), matrix)
    expected_post = pre + float(alpha) * (donor - pre)
    updated_fp32 = value + (expected_post - pre) @ matrix.T
    record = _coordinate_record(value, updated_fp32, matrix, expected_post)
    payload = {
        "basis": asdict(describe_basis(source, target)),
        "coordinates": asdict(record),
        "donor_source_coordinate": donor[..., 0].detach().cpu().reshape(-1).tolist(),
        "donor_target_coordinate": donor[..., 1].detach().cpu().reshape(-1).tolist(),
    }
    return updated_fp32.to(hidden.dtype), payload


def resolve_position_mask(
    mask: str,
    *,
    sequence_length: int,
    argument_positions: Sequence[int],
) -> list[int]:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    arguments = sorted({int(position) for position in argument_positions})
    if not arguments or arguments[0] < 0 or arguments[-1] >= sequence_length:
        raise ValueError("argument positions must be a non-empty in-range span")
    masks = {
        "final_prompt_position": [sequence_length - 1],
        "argument_token_position": arguments,
        "argument_through_end": list(range(arguments[0], sequence_length)),
        "all_prompt_positions": list(range(sequence_length)),
    }
    try:
        return masks[mask]
    except KeyError as exc:
        raise ValueError(f"unknown position mask: {mask}") from exc


def reconstruction_fraction(
    current_oriented: float,
    clean_oriented: float,
    installed_oriented: float,
    *,
    minimum_denominator: float = 1e-6,
) -> float | None:
    denominator = clean_oriented - installed_oriented
    if abs(denominator) < minimum_denominator:
        return None
    return (current_oriented - installed_oriented) / denominator
