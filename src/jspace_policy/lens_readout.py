"""Candidate unembedding parity for the pinned jlens HF adapter.

This module uses the same final norm, head dtype, bias and softcap as
jlens.hf at 581d398613e5602a5af361e1c34d3a92ea82ba8e. Its private adapter
attributes are intentional and must be revalidated on an upstream upgrade.
"""
from __future__ import annotations

from typing import Any


def selected_unembed(adapter: Any, residual: Any, token_ids: list[int]) -> Any:
    """Apply the reference unembedding pipeline without materializing all logits."""
    import torch
    import torch.nn.functional as functional

    head = adapter._lm_head
    normalized = adapter._final_norm(
        residual.to(device=head.weight.device, dtype=head.weight.dtype)
    )
    bias = None if head.bias is None else head.bias[token_ids]
    logits = functional.linear(normalized, head.weight[token_ids], bias)
    if adapter._logit_softcap is not None:
        cap = adapter._logit_softcap
        logits = cap * torch.tanh(logits / cap)
    return logits


def check_selected_unembed(adapter: Any, states: Any, token_ids: list[int]) -> None:
    """Refuse zero-only validation and compare nonzero states to the reference.

    BF16/FP16 GEMM shapes can differ numerically. The check uses PyTorch's
    dtype-specific assert_close defaults; numerical intervention identity is a
    separate, stricter instrument gate. This checks scoring, not causal efficacy.
    """
    import torch

    if states.numel() == 0 or not bool(torch.isfinite(states).all()):
        raise ValueError('parity states must be nonempty and finite')
    if not bool(states.abs().max() > 0):
        raise ValueError('zero-only states cannot validate normalization parity')
    reference = adapter.unembed(states)[..., token_ids]
    selected = selected_unembed(adapter, states, token_ids)
    torch.testing.assert_close(selected, reference)
