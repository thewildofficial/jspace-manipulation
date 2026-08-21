from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch

FP32_SWAP_ATOL = 2e-5
FP32_SWAP_RTOL = 2e-5


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class SwapDiagnostics:
    geometry_eligible: bool
    source_norm: float
    target_norm: float
    cosine: float | None
    condition_number: float | None
    coordinate_target_error: float
    delta_rms_ratio_mean: float
    delta_rms_ratio_max: float


def _basis(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    import torch

    if source.ndim != 1 or target.ndim != 1 or source.shape != target.shape:
        raise ValueError("source and target must be same-width vectors")
    matrix = torch.stack([source.float(), target.float()], dim=1)
    if int(torch.linalg.matrix_rank(matrix).item()) != 2:
        raise ValueError("source/target basis is rank deficient")
    return matrix


def projected_coordinates(hidden: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    import torch

    return hidden.float() @ torch.linalg.pinv(matrix).T


def coordinate_swap_fp32(
    hidden: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float,
) -> tuple[torch.Tensor, SwapDiagnostics]:
    """Apply the paper's two-coordinate swap, computing all geometry in fp32.

    Identity swaps deliberately take the no-op path. They validate hook parity but
    are not eligible for basis-conditioning summaries because ``[v, v]`` is rank
    deficient by construction.
    """
    import torch

    original_dtype = hidden.dtype
    value = hidden.float()
    source_fp32 = source.float().to(value.device)
    target_fp32 = target.float().to(value.device)
    source_norm = float(source_fp32.norm().detach().cpu())
    target_norm = float(target_fp32.norm().detach().cpu())
    if torch.equal(source_fp32, target_fp32):
        return hidden.clone(), SwapDiagnostics(
            geometry_eligible=False,
            source_norm=source_norm,
            target_norm=target_norm,
            cosine=None,
            condition_number=None,
            coordinate_target_error=0.0,
            delta_rms_ratio_mean=0.0,
            delta_rms_ratio_max=0.0,
        )

    matrix = _basis(source_fp32, target_fp32).to(value.device)
    pre = projected_coordinates(value, matrix)
    expected = pre + float(alpha) * (pre.flip(-1) - pre)
    updated = value + (expected - pre) @ matrix.T
    observed = projected_coordinates(updated, matrix)
    error = float((observed - expected).abs().max().detach().cpu())
    if not torch.allclose(
        observed,
        expected,
        atol=FP32_SWAP_ATOL,
        rtol=FP32_SWAP_RTOL,
    ):
        raise AssertionError(f"coordinate target error {error:.6g} exceeds tolerance")
    delta = updated - value
    denominator = value.square().mean(dim=-1).sqrt().clamp_min(1e-12)
    ratios = delta.square().mean(dim=-1).sqrt() / denominator
    cosine = torch.nn.functional.cosine_similarity(matrix[:, 0], matrix[:, 1], dim=0)
    return updated.to(original_dtype), SwapDiagnostics(
        geometry_eligible=True,
        source_norm=source_norm,
        target_norm=target_norm,
        cosine=float(cosine.detach().cpu()),
        condition_number=float(torch.linalg.cond(matrix).detach().cpu()),
        coordinate_target_error=error,
        delta_rms_ratio_mean=float(ratios.mean().detach().cpu()),
        delta_rms_ratio_max=float(ratios.max().detach().cpu()),
    )


def matched_random_swap_fp32(
    hidden: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return a deterministic random-basis swap matched to semantic delta per row.

    The random basis is explicitly projected out of the semantic two-vector span.
    Its swap delta is then rescaled independently for every patched activation so
    that its L2 norm equals the semantic swap delta at the same hook.
    """
    import torch

    value = hidden.float()
    semantic, semantic_diagnostics = coordinate_swap_fp32(value, source, target, alpha=alpha)
    semantic_delta = semantic.float() - value
    semantic_norm = semantic_delta.norm(dim=-1, keepdim=True)
    semantic_matrix = _basis(source.float(), target.float()).to(value.device)
    projector = semantic_matrix @ torch.linalg.pinv(semantic_matrix)

    random_basis: torch.Tensor | None = None
    for attempt in range(8):
        generator = torch.Generator(device="cpu").manual_seed(int(seed) + attempt)
        sample = torch.randn((value.shape[-1], 2), generator=generator, dtype=torch.float32).to(
            value.device
        )
        sample = sample - projector @ sample
        q, _ = torch.linalg.qr(sample, mode="reduced")
        if int(torch.linalg.matrix_rank(q).item()) == 2:
            random_basis = q
            break
    if random_basis is None:
        raise RuntimeError("could not construct deterministic random control basis")

    random_coords = projected_coordinates(value, random_basis)
    raw_delta = float(alpha) * (random_coords.flip(-1) - random_coords) @ random_basis.T
    raw_norm = raw_delta.norm(dim=-1, keepdim=True)
    scale = torch.where(
        semantic_norm <= 1e-12,
        torch.zeros_like(semantic_norm),
        semantic_norm / raw_norm.clamp_min(1e-12),
    )
    matched_delta = raw_delta * scale
    updated = value + matched_delta
    match_error = float(
        (matched_delta.norm(dim=-1) - semantic_norm.squeeze(-1)).abs().max().detach().cpu()
    )
    denominator = value.square().mean(dim=-1).sqrt().clamp_min(1e-12)
    ratios = matched_delta.square().mean(dim=-1).sqrt() / denominator
    return updated.to(hidden.dtype), {
        "semantic": asdict(semantic_diagnostics),
        "random_seed": int(seed),
        "delta_l2_match_max_abs_error": match_error,
        "delta_rms_ratio_mean": float(ratios.mean().detach().cpu()),
        "delta_rms_ratio_max": float(ratios.max().detach().cpu()),
    }


def balanced_unrelated_target(arguments: Sequence[str], source: str, target: str) -> str:
    """Map each target to the next member of the source-specific target cycle."""
    alternatives = [value for value in arguments if value != source]
    if target not in alternatives:
        raise ValueError("target must differ from source and occur in arguments")
    index = alternatives.index(target)
    return alternatives[(index + 1) % len(alternatives)]


def resolve_position_mask(
    mask: str, *, sequence_length: int, argument_positions: Sequence[int]
) -> list[int]:
    arguments = sorted({int(value) for value in argument_positions})
    if sequence_length <= 0 or not arguments:
        raise ValueError("sequence and argument positions must be non-empty")
    if arguments[0] < 0 or arguments[-1] >= sequence_length:
        raise ValueError("argument positions are outside the prompt")
    masks = {
        "all_prompt_positions": list(range(sequence_length)),
        "all_except_final_position": list(range(max(sequence_length - 1, 0))),
        "argument_through_end": list(range(arguments[0], sequence_length)),
        "final_prompt_position": [sequence_length - 1],
        "argument_token_position": arguments,
    }
    try:
        return masks[mask]
    except KeyError as exc:
        raise ValueError(f"unknown position mask: {mask}") from exc


def select_first_feasible_corpus(
    candidates: Mapping[str, Any], behavior_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Select the first frozen 4-argument x 4-function block per category.

    Ordering comes entirely from ``fresh_candidates.json``. A cell is feasible
    only when tokenizer checks pass and the clean source answer is top-1.
    """
    by_cell = {
        (str(row["category"]), str(row["function"]), str(row["argument"])): row
        for row in behavior_rows
    }
    frozen_categories: list[dict[str, Any]] = []
    for category in candidates["categories"]:
        category_name = str(category["name"])
        arguments = [str(value) for value in category["arguments"]]
        functions = [str(value["name"]) for value in category["functions"]]
        selected: tuple[tuple[str, ...], tuple[str, ...]] | None = None
        for argument_set in itertools.combinations(arguments, 4):
            for function_set in itertools.combinations(functions, 4):
                cells = [
                    by_cell.get((category_name, function, argument))
                    for function in function_set
                    for argument in argument_set
                ]
                if any(cell is None for cell in cells):
                    continue
                if not all(
                    bool(cell["tokenization_eligible"]) and bool(cell["clean_correct"])
                    for cell in cells
                    if cell is not None
                ):
                    continue
                answers_distinct = all(
                    len(
                        {
                            int(by_cell[(category_name, function, argument)]["answer_token_id"])
                            for argument in argument_set
                        }
                    )
                    == 4
                    for function in function_set
                )
                if answers_distinct:
                    selected = (argument_set, function_set)
                    break
            if selected is not None:
                break
        if selected is None:
            raise RuntimeError(f"no feasible 4x4 block for category {category_name}")

        argument_set, function_set = selected
        function_lookup = {str(value["name"]): value for value in category["functions"]}
        frozen_functions: list[dict[str, Any]] = []
        for function in function_set:
            definition = function_lookup[function]
            cell_records = {
                argument: dict(by_cell[(category_name, function, argument)])
                for argument in argument_set
            }
            frozen_functions.append(
                {
                    "name": function,
                    "template": definition["template"],
                    "answers": {
                        argument: definition["answers"][argument] for argument in argument_set
                    },
                    "cells": cell_records,
                }
            )
        frozen_categories.append(
            {
                "name": category_name,
                "arguments": list(argument_set),
                "functions": frozen_functions,
            }
        )

    frozen: dict[str, Any] = {
        "schema_version": 1,
        "study_id": candidates["study_id"],
        "status": "behavior_only_frozen_interventions_unopened",
        "selection_rule": "first_feasible_ordered_4_arguments_x_4_functions",
        "candidate_sha256": canonical_sha256(candidates),
        "categories": frozen_categories,
    }
    frozen["content_sha256"] = canonical_sha256(frozen)
    return frozen


def cluster_bootstrap_mean(
    values: Sequence[float], clusters: Sequence[str], *, draws: int, seed: int
) -> tuple[float, float]:
    if len(values) != len(clusters) or not values:
        raise ValueError("values and clusters must have equal non-zero length")
    grouped: dict[str, list[float]] = {}
    for value, cluster in zip(values, clusters, strict=True):
        grouped.setdefault(cluster, []).append(float(value))
    keys = sorted(grouped)
    cluster_means = np.array([np.mean(grouped[key]) for key in keys], dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(cluster_means, size=(int(draws), len(keys)), replace=True)
    lower, upper = np.quantile(samples.mean(axis=1), [0.025, 0.975])
    return float(lower), float(upper)
