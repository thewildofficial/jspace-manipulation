from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager

import torch
from torch import nn


def j_lens_vector(
    jacobian: torch.Tensor,
    unembedding_weight: torch.Tensor,
    token_id: int,
    *,
    unit_norm: bool = True,
) -> torch.Tensor:
    """Return row ``token_id`` of ``W_U J`` in layer-residual coordinates."""
    if jacobian.ndim != 2 or unembedding_weight.ndim != 2:
        raise ValueError("jacobian and unembedding weight must be matrices")
    if jacobian.shape[0] != unembedding_weight.shape[1]:
        raise ValueError("incompatible Jacobian and unembedding dimensions")
    vector = jacobian.T.float() @ unembedding_weight[token_id].float()
    if unit_norm:
        norm = vector.norm()
        if norm <= 0:
            raise ValueError("cannot normalize a zero J-lens vector")
        vector = vector / norm
    return vector


def steer(hidden: torch.Tensor, direction: torch.Tensor, alpha: float) -> torch.Tensor:
    return hidden + alpha * direction.to(device=hidden.device, dtype=hidden.dtype)


def ablate(hidden: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    unit = direction.float() / direction.float().norm().clamp_min(1e-12)
    unit = unit.to(device=hidden.device, dtype=hidden.dtype)
    coordinate = torch.einsum("...d,d->...", hidden, unit)
    return hidden - coordinate.unsqueeze(-1) * unit


def coordinate_swap(
    hidden: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Swap two local lens coordinates while preserving the orthogonal component."""
    original_dtype = hidden.dtype
    h = hidden.float()
    matrix = torch.stack([source.float(), target.float()], dim=1).to(h.device)  # [d, 2]
    coords = torch.einsum("kd,...d->...k", torch.linalg.pinv(matrix), h)
    swapped = coords.flip(-1)
    delta = torch.einsum("dk,...k->...d", matrix, alpha * (swapped - coords))
    return (h + delta).to(original_dtype)


class ResidualIntervention(AbstractContextManager["ResidualIntervention"]):
    """Temporarily modify selected sequence positions at one residual block."""

    def __init__(
        self,
        blocks: Sequence[nn.Module],
        layer: int,
        transform: Callable[[torch.Tensor], torch.Tensor],
        *,
        positions: Sequence[int] = (-1,),
    ) -> None:
        self.block = blocks[layer]
        self.transform = transform
        self.positions = tuple(positions)
        self.handle: torch.utils.hooks.RemovableHandle | None = None

    def _hook(self, module: nn.Module, inputs: object, output: object) -> object:
        tensor = output if torch.is_tensor(output) else output[0]
        updated = tensor.clone()
        resolved = [p if p >= 0 else tensor.shape[1] + p for p in self.positions]
        updated[:, resolved, :] = self.transform(updated[:, resolved, :])
        if torch.is_tensor(output):
            return updated
        return (updated, *output[1:])

    def __enter__(self) -> ResidualIntervention:
        self.handle = self.block.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc: object) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
