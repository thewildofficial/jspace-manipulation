"""Bounded Modal GPU entry point for the V2 instrumentation gate (H0).

Run from the repository root:

    modal run modal_v2.py::h0

The job first maps a task-independent workspace band, then runs numerical,
synthetic, directional, and Anthropic flexible-generalization controls. It
does not run any strategic-reporting task.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

from jspace_policy.budget import admit_run, append_ledger, estimate_cost

APP_NAME = "jspace-v2-h0"
MODEL_ID = "Qwen/Qwen3.6-27B"
MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
LENS_FILENAME = (
    "qwen3.6-27b/jlens/Salesforce-wikitext/"
    "Qwen3.6-27B_jacobian_lens_n1000.pt"
)
JLENS_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
SEED = 1729

app = modal.App(APP_NAME)
cache = modal.Volume.from_name("jspace-hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .uv_pip_install(
        "torch>=2.8",
        "transformers>=5.5",
        "huggingface_hub>=0.34",
        f"git+https://github.com/anthropics/jacobian-lens.git@{JLENS_COMMIT}",
    )
    .env({"HF_HOME": "/cache/huggingface", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir("src/jspace_policy", remote_path="/root/jspace_policy")
)


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


def _one_token_id(tokenizer: Any, word: str, *, continuation: bool = True) -> int:
    variants = [f" {word}", word] if continuation else [word, f" {word}"]
    for variant in variants:
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if len(ids) == 1:
            return int(ids[0])
    raise ValueError(f"{word!r} has no accepted one-token form")


def _continuation_id(tokenizer: Any, prompt: str, answer: str) -> int:
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    for suffix in (f" {answer}", answer):
        full = tokenizer.encode(prompt + suffix, add_special_tokens=False)
        if full[: len(prefix)] == prefix and len(full) == len(prefix) + 1:
            return int(full[-1])
    raise ValueError(f"{answer!r} is not one token after prompt {prompt!r}")


def _excess_kurtosis(logits: Any) -> Any:
    centered = logits.float() - logits.float().mean(dim=-1, keepdim=True)
    variance = centered.square().mean(dim=-1).clamp_min(1e-12)
    return centered.pow(4).mean(dim=-1) / variance.square() - 3.0


def _centered_cosine(left: Any, right: Any) -> Any:
    import torch

    left = left.float() - left.float().mean(dim=-1, keepdim=True)
    right = right.float() - right.float().mean(dim=-1, keepdim=True)
    return torch.nn.functional.cosine_similarity(left, right, dim=-1)


def _select_workspace_band(diagnostics: list[dict[str, object]], config: dict) -> list[int]:
    import torch

    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in diagnostics:
        grouped[int(row["layer"])].append(row)
    summaries: list[dict[str, float]] = []
    for layer, rows in sorted(grouped.items()):
        summaries.append(
            {
                "layer": float(layer),
                "layer_fraction": float(rows[0]["layer_fraction"]),
                "mean_excess_kurtosis": sum(
                    float(row["excess_kurtosis"]) for row in rows
                )
                / len(rows),
                "mean_jlens_logitlens_cosine": sum(
                    float(row["jlens_logitlens_cosine"]) for row in rows
                )
                / len(rows),
            }
        )
    low, high = config["selection_rule"]["candidate_normalized_depth"]
    candidates = [row for row in summaries if low <= row["layer_fraction"] <= high]
    if len(candidates) < int(config["selection_rule"]["band_width_layers"]):
        raise RuntimeError("too few fitted layers in the preregistered candidate interval")
    kurtosis = torch.tensor([row["mean_excess_kurtosis"] for row in candidates])
    motor = torch.tensor([row["mean_jlens_logitlens_cosine"] for row in candidates])
    kurtosis_z = (kurtosis - kurtosis.mean()) / kurtosis.std().clamp_min(1e-12)
    motor_z = (motor - motor.mean()) / motor.std().clamp_min(1e-12)
    for index, row in enumerate(candidates):
        row["selection_score"] = float(kurtosis_z[index] - 0.5 * motor_z[index])
    width = int(config["selection_rule"]["band_width_layers"])
    windows: list[tuple[float, int, list[int]]] = []
    for start in range(len(candidates) - width + 1):
        window = candidates[start : start + width]
        layers = [int(row["layer"]) for row in window]
        if layers != list(range(layers[0], layers[0] + width)):
            continue
        score = sum(row["selection_score"] for row in window) / width
        windows.append((score, -layers[0], layers))
    if not windows:
        raise RuntimeError("no contiguous band satisfies the frozen selection rule")
    selected = max(windows)[2]
    summary_by_layer = {int(row["layer"]): row for row in summaries}
    score_by_layer = {int(row["layer"]): row for row in candidates}
    for row in diagnostics:
        layer = int(row["layer"])
        row.update(summary_by_layer[layer])
        row["selection_score"] = score_by_layer.get(layer, {}).get("selection_score")
        row["selected"] = layer in selected
    return selected


def _workspace_mapping(model: Any, lens: Any, config: dict) -> tuple[list[int], list[dict]]:
    import torch
    from jlens import ActivationRecorder

    rows: list[dict[str, object]] = []
    minimum_position = int(config["minimum_token_position"])
    n_positions = int(config["positions_per_prompt"])
    final_layer = model.n_layers - 1
    for prompt_index, prompt in enumerate(config["neutral_prompts"]):
        input_ids = model.encode(prompt, max_length=512)
        seq_len = int(input_ids.shape[1])
        valid = list(range(minimum_position, seq_len))
        if not valid:
            rows.append(
                {
                    "record_type": "excluded_prompt",
                    "prompt_index": prompt_index,
                    "prompt_sha256": _prompt_sha256(prompt),
                    "reason": "no_position_at_or_after_fitting_floor",
                    "sequence_length": seq_len,
                }
            )
            continue
        indices = torch.linspace(0, len(valid) - 1, steps=min(n_positions, len(valid)))
        positions = sorted({valid[int(round(float(index)))] for index in indices})
        with ActivationRecorder(
            model.layers, at=[*lens.source_layers, final_layer]
        ) as recorder:
            model.forward(input_ids)
        for layer in lens.source_layers:
            residual = recorder.activations[layer][0, positions].detach().float()
            jacobian = lens.jacobians[layer].to(residual.device)
            transported = residual @ jacobian.T
            del jacobian
            jlens_logits = model.unembed(transported).float()
            logit_lens_logits = model.unembed(residual).float()
            kurtosis = _excess_kurtosis(jlens_logits)
            cosine = _centered_cosine(jlens_logits, logit_lens_logits)
            agreement = jlens_logits.argmax(-1).eq(logit_lens_logits.argmax(-1))
            top_ids = jlens_logits.argmax(-1)
            for offset, position in enumerate(positions):
                rows.append(
                    {
                        "record_type": "workspace_diagnostic",
                        "prompt_index": prompt_index,
                        "prompt_sha256": _prompt_sha256(prompt),
                        "sequence_length": seq_len,
                        "position": position,
                        "position_token_id": int(input_ids[0, position]),
                        "layer": int(layer),
                        "layer_fraction": float(layer / max(model.n_layers - 1, 1)),
                        "excess_kurtosis": float(kurtosis[offset].cpu()),
                        "jlens_logitlens_cosine": float(cosine[offset].cpu()),
                        "jlens_logitlens_top1_agreement": bool(agreement[offset].cpu()),
                        "jlens_top1_token_id": int(top_ids[offset].cpu()),
                    }
                )
            del residual, transported, jlens_logits, logit_lens_logits
    measured = [row for row in rows if row["record_type"] == "workspace_diagnostic"]
    selected = _select_workspace_band(measured, config)
    return selected, rows


def _synthetic_swap_check() -> dict[str, object]:
    import torch

    from jspace_policy.interventions import coordinate_swap

    hidden = torch.tensor([[2.0, -3.0, 7.0]])
    source = torch.tensor([1.0, 0.0, 0.0])
    target = torch.tensor([0.0, 1.0, 0.0])
    expected = torch.tensor([[-3.0, 2.0, 7.0]])
    actual = coordinate_swap(hidden, source, target)
    return {
        "input": hidden.tolist(),
        "expected": expected.tolist(),
        "actual": actual.tolist(),
        "max_abs_error": float((actual - expected).abs().max()),
    }


def _parity_checks(model: Any, lens: Any, selected_band: list[int]) -> dict[str, object]:
    import torch
    from jlens import ActivationRecorder

    from jspace_policy.interventions import j_lens_vector

    layer = selected_band[len(selected_band) // 2]
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    residual = torch.randn(3, lens.d_model, generator=generator).to("cuda")
    upstream = lens.transport(residual, layer)
    local = residual @ lens.jacobians[layer].to(residual.device).T
    transport_delta = upstream - local
    upstream_logits = model.unembed(upstream).float()
    local_logits = model.unembed(local).float()
    top_k = 10
    upstream_top = upstream_logits.topk(top_k).indices
    local_top = local_logits.topk(top_k).indices

    prompt = "Fact: The currency used in the country shaped like a boot is"
    upstream_readout, _, input_ids = lens.apply(
        model, prompt, layers=[layer], positions=[-1]
    )
    with ActivationRecorder(model.layers, at=[layer]) as recorder:
        model.forward(model.encode(prompt, max_length=512))
    activation = recorder.activations[layer][0, -1].detach().float()
    manual_readout = model.unembed(
        activation @ lens.jacobians[layer].to(activation.device).T
    ).float().cpu()
    prompt_delta = upstream_readout[layer][0] - manual_readout

    token_id = _one_token_id(model.tokenizer, "China")
    unembedding = model._hf_model.lm_head.weight.detach()
    jacobian_gpu = lens.jacobians[layer].to(unembedding.device)
    helper_vector = j_lens_vector(
        jacobian_gpu, unembedding, token_id
    )
    formula_vector = jacobian_gpu.T @ unembedding[token_id].float()
    formula_vector = formula_vector / formula_vector.norm()
    return {
        "layer": layer,
        "random_residual": {
            "max_abs_error": float(transport_delta.abs().max().cpu()),
            "mean_abs_error": float(transport_delta.abs().mean().cpu()),
            "cosine": float(
                torch.nn.functional.cosine_similarity(
                    upstream.flatten(), local.flatten(), dim=0
                ).cpu()
            ),
            "top_k": top_k,
            "top_k_exact_agreement": float(upstream_top.eq(local_top).float().mean().cpu()),
        },
        "actual_prompt": {
            "prompt_sha256": _prompt_sha256(prompt),
            "sequence_length": int(input_ids.shape[1]),
            "max_abs_error": float(prompt_delta.abs().max()),
            "mean_abs_error": float(prompt_delta.abs().mean()),
            "cosine": float(
                torch.nn.functional.cosine_similarity(
                    upstream_readout[layer][0], manual_readout, dim=0
                )
            ),
            "top_k_exact_agreement": float(
                upstream_readout[layer][0]
                .topk(top_k)
                .indices.eq(manual_readout.topk(top_k).indices)
                .float()
                .mean()
            ),
        },
        "direction_formula": {
            "token_id": token_id,
            "max_abs_error": float((helper_vector - formula_vector).abs().max()),
            "cosine": float(
                torch.nn.functional.cosine_similarity(
                    helper_vector, formula_vector, dim=0
                )
            ),
        },
    }


def _direction_check(model: Any, lens: Any, selected_band: list[int]) -> list[dict]:
    import torch

    from jspace_policy.interventions import ResidualIntervention, j_lens_vector

    layer = selected_band[len(selected_band) // 2]
    prompt = "The capital of France is the city of"
    input_ids = model.encode(prompt, max_length=512)
    captured: dict[str, Any] = {}

    def capture(module: Any, inputs: object, output: object) -> None:
        tensor = output if torch.is_tensor(output) else output[0]
        captured["residual"] = tensor.detach()[0, -1].float()

    handle = model.layers[layer].register_forward_hook(capture)
    with torch.inference_mode():
        baseline_logits = model._hf_model(input_ids).logits[0, -1].float()
    handle.remove()
    residual = captured["residual"]
    token_id = _one_token_id(model.tokenizer, "China")
    vector = j_lens_vector(
        lens.jacobians[layer].to(residual.device),
        model._hf_model.lm_head.weight.detach(),
        token_id,
    )
    transported = residual @ lens.jacobians[layer].to(residual.device).T
    baseline_predicted = model.unembed(transported).float()
    rows: list[dict[str, object]] = []
    for relative_scale in (1e-4, 1e-3, 1e-2):
        epsilon = relative_scale * float(residual.norm())
        delta = epsilon * vector
        predicted = model.unembed(
            (residual + delta) @ lens.jacobians[layer].to(residual.device).T
        ).float()

        def transform(value: Any, delta: Any = delta) -> Any:
            return value + delta.to(value.dtype)

        with torch.inference_mode(), ResidualIntervention(
            model.layers, layer, transform, positions=(-1,)
        ):
            actual = model._hf_model(input_ids).logits[0, -1].float()
        predicted_delta = predicted - baseline_predicted
        actual_delta = actual - baseline_logits
        rows.append(
            {
                "layer": layer,
                "prompt_sha256": _prompt_sha256(prompt),
                "target_token_id": token_id,
                "relative_scale": relative_scale,
                "epsilon": epsilon,
                "predicted_target_logit_delta": float(predicted_delta[token_id].cpu()),
                "actual_target_logit_delta": float(actual_delta[token_id].cpu()),
                "direction_agrees": bool(
                    float(predicted_delta[token_id].cpu())
                    * float(actual_delta[token_id].cpu())
                    > 0
                ),
                "full_vocab_delta_cosine": float(
                    torch.nn.functional.cosine_similarity(
                        predicted_delta, actual_delta, dim=0
                    ).cpu()
                ),
            }
        )
    return rows


def _precompute_directions(
    model: Any, lens: Any, layers: list[int], token_ids: list[int]
) -> dict[int, dict[int, Any]]:
    weights = model._hf_model.lm_head.weight.detach()[token_ids].float()
    result: dict[int, dict[int, Any]] = {}
    for layer in layers:
        jacobian = lens.jacobians[layer].to(weights.device)
        vectors = weights @ jacobian
        vectors = vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        result[layer] = {
            token_id: vectors[index].detach()
            for index, token_id in enumerate(token_ids)
        }
        del jacobian
    return result


def _apply_basis_swap(hidden: Any, source: Any, target: Any, alpha: float) -> tuple[Any, Any]:
    import torch

    original_dtype = hidden.dtype
    value = hidden.float()
    matrix = torch.stack([source.float(), target.float()], dim=1)
    coords = value @ torch.linalg.pinv(matrix).T
    delta = (alpha * (coords.flip(-1) - coords)) @ matrix.T
    return (value + delta).to(original_dtype), delta


def _make_batched_swap_transform(
    layer: int,
    variants: list[dict[str, object]],
    directions: dict[int, dict[int, Any]],
    arg_token_ids: dict[str, int],
    source_arg: str,
    alpha: float,
    diagnostics: dict[int, dict[int, dict[str, float]]],
):
    def transform(value: Any) -> Any:
        for variant_index, variant in enumerate(variants):
            if layer not in variant["layers"]:
                continue
            positions: Any = (
                slice(None) if variant["all_positions"] else slice(-1, None)
            )
            before = value[variant_index, positions, :].clone()
            source = directions[layer][arg_token_ids[source_arg]]
            target_arg = str(variant["target_arg"])
            target = directions[layer][arg_token_ids[target_arg]]
            after, delta = _apply_basis_swap(before, source, target, alpha)
            value[variant_index, positions, :] = after
            before_float = before.float()
            diagnostics[variant_index][layer] = {
                "delta_l2": float(delta.norm(dim=-1).mean().detach().cpu()),
                "residual_rms": float(
                    before_float.square().mean(dim=-1).sqrt().mean().detach().cpu()
                ),
                "delta_rms_ratio": float(
                    (
                        delta.square().mean(dim=-1).sqrt()
                        / before_float.square()
                        .mean(dim=-1)
                        .sqrt()
                        .clamp_min(1e-12)
                    )
                    .mean()
                    .detach()
                    .cpu()
                ),
            }
        return value

    return transform


def _flexible_generalization(
    model: Any,
    lens: Any,
    selected_band: list[int],
    config: dict,
) -> tuple[list[dict], dict[str, object]]:
    import torch

    from jspace_policy.interventions import MultiLayerResidualIntervention

    center_layer = selected_band[len(selected_band) // 2]
    all_layers = sorted({center_layer, *selected_band})
    categories = config["data"]["categories"]
    arg_token_ids: dict[str, int] = {}
    tokenization_exclusions: list[dict[str, str]] = []
    for category in categories:
        for arg in category["args"]:
            try:
                arg_token_ids[arg] = _one_token_id(model.tokenizer, arg)
            except ValueError as exc:
                tokenization_exclusions.append({"argument": arg, "reason": str(exc)})
    directions = _precompute_directions(
        model, lens, all_layers, sorted(set(arg_token_ids.values()))
    )
    topologies = config["topologies"]
    rows: list[dict[str, object]] = []
    alpha = float(config["alpha"])
    for category in categories:
        category_name = category["name"]
        for function in category["funcs"]:
            function_name = function["name"]
            for source_arg in category["args"]:
                prompt = function["template"].format(arg=source_arg)
                input_ids = model.encode(prompt, max_length=512)
                try:
                    source_answer_id = _continuation_id(
                        model.tokenizer, prompt, function["answers"][source_arg]
                    )
                except ValueError as exc:
                    tokenization_exclusions.append(
                        {
                            "category": category_name,
                            "function": function_name,
                            "argument": source_arg,
                            "reason": str(exc),
                        }
                    )
                    continue
                if source_arg not in arg_token_ids:
                    continue
                with torch.inference_mode():
                    baseline_logits = model._hf_model(input_ids).logits[0, -1].float()
                    baseline_log_probs = baseline_logits.log_softmax(-1)
                baseline_top1 = int(baseline_logits.argmax().cpu())
                variants: list[dict[str, object]] = []
                for target_arg in category["args"]:
                    if target_arg == source_arg or target_arg not in arg_token_ids:
                        continue
                    try:
                        target_answer_id = _continuation_id(
                            model.tokenizer, prompt, function["answers"][target_arg]
                        )
                    except ValueError as exc:
                        tokenization_exclusions.append(
                            {
                                "category": category_name,
                                "function": function_name,
                                "argument": source_arg,
                                "target_argument": target_arg,
                                "reason": str(exc),
                            }
                        )
                        continue
                    for topology in topologies:
                        variants.append(
                            {
                                "target_arg": target_arg,
                                "target_answer_id": target_answer_id,
                                "topology": topology,
                                "layers": (
                                    selected_band
                                    if topology.startswith("workspace_band")
                                    else [center_layer]
                                ),
                                "all_positions": topology.endswith("all_positions"),
                            }
                        )
                if not variants:
                    continue
                batch_ids = input_ids.repeat(len(variants), 1)
                intervention_diagnostics: dict[
                    int, dict[int, dict[str, float]]
                ] = defaultdict(dict)
                transforms = {
                    layer: _make_batched_swap_transform(
                        layer,
                        variants,
                        directions,
                        arg_token_ids,
                        source_arg,
                        alpha,
                        intervention_diagnostics,
                    )
                    for layer in all_layers
                }
                with torch.inference_mode(), MultiLayerResidualIntervention(
                    model.layers, transforms
                ):
                    intervened_logits = model._hf_model(batch_ids).logits[:, -1].float()
                intervened_log_probs = intervened_logits.log_softmax(-1)
                for variant_index, variant in enumerate(variants):
                    target_id = int(variant["target_answer_id"])
                    baseline_target_vs_source = float(
                        (
                            baseline_log_probs[target_id]
                            - baseline_log_probs[source_answer_id]
                        ).cpu()
                    )
                    intervened_target_vs_source = float(
                        (
                            intervened_log_probs[variant_index, target_id]
                            - intervened_log_probs[variant_index, source_answer_id]
                        ).cpu()
                    )
                    layer_diagnostics = intervention_diagnostics[variant_index]
                    rows.append(
                        {
                            "trial_id": uuid.uuid4().hex,
                            "category": category_name,
                            "function": function_name,
                            "prompt": prompt,
                            "prompt_sha256": _prompt_sha256(prompt),
                            "prompt_token_ids": [int(token) for token in input_ids[0].cpu()],
                            "sequence_length": int(input_ids.shape[1]),
                            "source_argument": source_arg,
                            "source_argument_token_id": arg_token_ids[source_arg],
                            "target_argument": variant["target_arg"],
                            "target_argument_token_id": arg_token_ids[
                                str(variant["target_arg"])
                            ],
                            "source_answer": function["answers"][source_arg],
                            "source_answer_token_id": source_answer_id,
                            "target_answer": function["answers"][str(variant["target_arg"])],
                            "target_answer_token_id": target_id,
                            "baseline_top1_token_id": baseline_top1,
                            "baseline_correct": baseline_top1 == source_answer_id,
                            "topology": variant["topology"],
                            "intervened_layers": variant["layers"],
                            "position_mask": (
                                "all_prompt_positions"
                                if variant["all_positions"]
                                else "final_prompt_position"
                            ),
                            "alpha": alpha,
                            "intervened_top1_token_id": int(
                                intervened_logits[variant_index].argmax().cpu()
                            ),
                            "swap_success": int(
                                intervened_logits[variant_index].argmax().cpu()
                            )
                            == target_id,
                            "baseline_target_vs_source_logodds": baseline_target_vs_source,
                            "intervened_target_vs_source_logodds": intervened_target_vs_source,
                            "target_logodds_gain": (
                                intervened_target_vs_source - baseline_target_vs_source
                            ),
                            "layer_diagnostics": {
                                str(layer): values
                                for layer, values in sorted(layer_diagnostics.items())
                            },
                            "mean_delta_rms_ratio": (
                                sum(
                                    value["delta_rms_ratio"]
                                    for value in layer_diagnostics.values()
                                )
                                / max(len(layer_diagnostics), 1)
                            ),
                        }
                    )
    summary_rows: list[dict[str, object]] = []
    for topology in topologies:
        part = [row for row in rows if row["topology"] == topology]
        eligible = [row for row in part if row["baseline_correct"]]
        summary_rows.append(
            {
                "topology": topology,
                "n_trials": len(part),
                "n_eligible": len(eligible),
                "baseline_accuracy": (
                    sum(bool(row["baseline_correct"]) for row in part) / max(len(part), 1)
                ),
                "conditional_swap_success_rate": (
                    sum(bool(row["swap_success"]) for row in eligible)
                    / max(len(eligible), 1)
                ),
                "mean_target_logodds_gain": (
                    sum(float(row["target_logodds_gain"]) for row in eligible)
                    / max(len(eligible), 1)
                ),
                "mean_delta_rms_ratio": (
                    sum(float(row["mean_delta_rms_ratio"]) for row in eligible)
                    / max(len(eligible), 1)
                ),
            }
        )
    return rows, {
        "center_layer": center_layer,
        "workspace_band": selected_band,
        "summary_by_topology": summary_rows,
        "tokenization_exclusions": tokenization_exclusions,
    }


def _evaluate_h0(
    model: Any,
    lens: Any,
    workspace_config: dict,
    flexible_config: dict,
) -> tuple[dict[str, object], list[dict], list[dict]]:
    selected_band, workspace_rows = _workspace_mapping(model, lens, workspace_config)
    synthetic = _synthetic_swap_check()
    parity = _parity_checks(model, lens, selected_band)
    direction = _direction_check(model, lens, selected_band)
    flexible_rows, flexible_summary = _flexible_generalization(
        model, lens, selected_band, flexible_config
    )
    topology = {
        row["topology"]: row for row in flexible_summary["summary_by_topology"]
    }
    reference = topology[flexible_config["primary_positive_control"]]
    local = topology[flexible_config["primary_weakened_control"]]
    pass_rule = flexible_config["h0_pass_rule"]
    instrument_rule = flexible_config["instrument_pass_rule"]
    parity_pass = (
        float(parity["random_residual"]["max_abs_error"])
        <= float(instrument_rule["random_transport_max_abs_error"])
        and float(parity["actual_prompt"]["max_abs_error"])
        <= float(instrument_rule["actual_prompt_readout_max_abs_error"])
        and float(parity["direction_formula"]["max_abs_error"])
        <= float(instrument_rule["direction_formula_max_abs_error"])
    )
    direction_pass = sum(bool(row["direction_agrees"]) for row in direction) >= int(
        instrument_rule["minimum_direction_agreements_of_three"]
    )
    synthetic_pass = float(synthetic["max_abs_error"]) <= float(
        instrument_rule["synthetic_swap_max_abs_error"]
    )
    flexible_pass = (
        int(reference["n_eligible"]) >= int(pass_rule["minimum_eligible_trials"])
        and float(reference["conditional_swap_success_rate"])
        >= float(pass_rule["minimum_reference_success_rate"])
        and float(reference["conditional_swap_success_rate"])
        - float(local["conditional_swap_success_rate"])
        >= float(pass_rule["minimum_reference_minus_local_success_rate"])
        and float(reference["mean_target_logodds_gain"])
        >= float(pass_rule["minimum_reference_target_logodds_gain"])
    )
    summary = {
        "workspace_band": selected_band,
        "synthetic_swap": synthetic,
        "parity": parity,
        "direction_check": direction,
        "flexible_generalization": flexible_summary,
        "gate": {
            "parity_pass": parity_pass,
            "synthetic_swap_pass": synthetic_pass,
            "direction_pass": direction_pass,
            "flexible_generalization_pass": flexible_pass,
            "h0_pass": parity_pass and synthetic_pass and direction_pass and flexible_pass,
        },
    }
    return summary, workspace_rows, flexible_rows


@app.function(
    image=image,
    volumes={"/cache": cache},
    cpu=4.0,
    memory=32768,
    gpu="A100-80GB",
    timeout=3600,
    max_containers=1,
    retries=0,
)
def run_h0_remote(
    workspace_config: dict,
    flexible_config: dict,
    git_commit: str,
    dirty_tree: bool,
) -> str:
    import torch
    import transformers
    from huggingface_hub import model_info
    from jlens import JacobianLens, from_hf

    started = time.perf_counter()
    torch.manual_seed(SEED)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION
    )
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).cuda()
    hf_model.eval()
    model = from_hf(hf_model, tokenizer)
    lens = JacobianLens.from_pretrained(
        LENS_REPO, filename=LENS_FILENAME, revision=LENS_REVISION
    )
    if lens.d_model != model.d_model:
        raise RuntimeError(f"lens width {lens.d_model} != model width {model.d_model}")
    summary, workspace_rows, flexible_rows = _evaluate_h0(
        model, lens, workspace_config, flexible_config
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    cache.commit()
    resolved = model_info(MODEL_ID, revision=MODEL_REVISION).sha
    run_id = uuid.uuid4().hex
    metadata = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
        "dirty_tree": dirty_tree,
        "model_id": MODEL_ID,
        "model_revision_requested": MODEL_REVISION,
        "model_revision_resolved": resolved,
        "tokenizer_revision": resolved,
        "lens_repo": LENS_REPO,
        "lens_filename": LENS_FILENAME,
        "lens_revision": LENS_REVISION,
        "lens_code_commit": JLENS_COMMIT,
        "lens_source_layers": lens.source_layers,
        "seed": SEED,
        "dtype": "bfloat16",
        "gpu_requested": "A100-80GB",
        "gpu_actual": torch.cuda.get_device_name(0),
        "torch_version": str(torch.__version__),
        "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda,
        "workspace_config_hash": _sha256_json(workspace_config),
        "flexible_config_hash": _sha256_json(flexible_config),
        "elapsed_seconds": elapsed,
    }
    result = {
        "metadata": metadata,
        "summary": summary,
        "workspace_rows": workspace_rows,
        "flexible_rows": flexible_rows,
    }
    return json.dumps(result, allow_nan=False)


def _write_jsonl(rows: list[dict], path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(metadata | row, sort_keys=True) + "\n")


@app.local_entrypoint()
def h0() -> None:
    """Run and download the complete bounded V2 instrumentation gate."""
    workspace_config = json.loads(Path("configs/v2/workspace_mapping.json").read_text())
    flexible_config = json.loads(
        Path("configs/v2/flexible_generalization_smoke.json").read_text()
    )
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    dirty_tree = bool(
        subprocess.run(
            ["git", "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    estimate = estimate_cost("A100-80GB", 1800, memory_gib=32.0)
    ledger = Path("artifacts/raw/cost_ledger.jsonl")
    admit_run(ledger, estimate)
    result = json.loads(
        run_h0_remote.remote(
            workspace_config, flexible_config, git_commit, dirty_tree
        )
    )
    metadata = result["metadata"]
    smoke_root = Path("results/v2_smoke_tests")
    workspace_root = Path("results/v2_workspace_mapping")
    smoke_root.joinpath("raw").mkdir(parents=True, exist_ok=True)
    workspace_root.joinpath("raw").mkdir(parents=True, exist_ok=True)
    smoke_root.joinpath("raw/h0_summary.json").write_text(
        json.dumps({"metadata": metadata, "summary": result["summary"]}, indent=2)
        + "\n"
    )
    _write_jsonl(
        result["flexible_rows"],
        smoke_root / "raw/flexible_generalization.jsonl",
        metadata,
    )
    _write_jsonl(
        result["workspace_rows"],
        workspace_root / "raw/workspace_diagnostics.jsonl",
        metadata,
    )
    workspace_manifest = metadata | {
        "selected_band": result["summary"]["workspace_band"],
        "selection_rule": workspace_config["selection_rule"],
        "n_rows": len(result["workspace_rows"]),
    }
    workspace_root.joinpath("workspace_band.json").write_text(
        json.dumps(workspace_manifest, indent=2, sort_keys=True) + "\n"
    )
    smoke_manifest = metadata | {
        "gate": result["summary"]["gate"],
        "n_flexible_rows": len(result["flexible_rows"]),
        "commands": ["modal run modal_v2.py::h0"],
    }
    smoke_root.joinpath("run_manifest.json").write_text(
        json.dumps(smoke_manifest, indent=2, sort_keys=True) + "\n"
    )
    workspace_root.joinpath("run_manifest.json").write_text(
        json.dumps(workspace_manifest, indent=2, sort_keys=True) + "\n"
    )
    measured = estimate_cost(
        "A100-80GB", float(metadata["elapsed_seconds"]), memory_gib=32.0
    )
    append_ledger(
        ledger,
        measured,
        run_id=str(metadata["run_id"]),
        stage="v2-h0-instrumentation",
    )
    print(
        json.dumps(
            {
                "run_id": metadata["run_id"],
                "elapsed_seconds": metadata["elapsed_seconds"],
                "recorded_cost_usd": measured.buffered_usd,
                "gate": result["summary"]["gate"],
                "workspace_band": result["summary"]["workspace_band"],
            },
            indent=2,
        )
    )
