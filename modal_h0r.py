"""Modal entry points for H0R causal-instrument recovery."""

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

MODEL_ID = "Qwen/Qwen3.6-27B"
MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
JLENS_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
LENS_FILENAME = (
    "qwen3.6-27b/jlens/Salesforce-wikitext/"
    "Qwen3.6-27B_jacobian_lens_n1000.pt"
)
SEED = 1729

app = modal.App("jspace-h0r")
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


@app.function(image=image, volumes={"/cache": cache}, cpu=2.0, memory=4096, timeout=900)
def tokenize_locked_controls() -> str:
    import transformers

    from jspace_policy.h0r import build_locked_controls

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION
    )
    controls = build_locked_controls(tokenizer)
    controls["created_at"] = datetime.now(UTC).isoformat()
    controls["model_id"] = MODEL_ID
    controls["tokenizer_revision"] = MODEL_REVISION
    payload = json.dumps(controls, sort_keys=True, separators=(",", ":"))
    controls["content_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    cache.commit()
    return json.dumps(controls, indent=2, sort_keys=True)


@app.local_entrypoint()
def freeze_controls() -> None:
    output = Path("configs/v2/h0r_locked_controls.json")
    if output.exists():
        raise RuntimeError(f"refusing to overwrite locked control: {output}")
    payload = tokenize_locked_controls.remote()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload + "\n")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    parsed = json.loads(payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "parent_commit": commit,
                "content_sha256": parsed["content_sha256"],
                "argument_trials": len(parsed["argument_control"]["trials"]),
                "intermediate_trials": len(parsed["intermediate_control"]["trials"]),
            },
            indent=2,
        )
    )


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


def _argument_positions(tokenizer: Any, prompt: str, argument: str) -> list[int]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    for surface in (argument, f" {argument}"):
        piece = tokenizer.encode(surface, add_special_tokens=False)
        for start in range(len(prompt_ids) - len(piece) + 1):
            if prompt_ids[start : start + len(piece)] == piece:
                return list(range(start, start + len(piece)))
    raise RuntimeError(f"argument span not found: {argument!r} in {prompt!r}")


def _token_id(tokenizer: Any, value: str) -> int:
    for surface in (f" {value}", value):
        ids = tokenizer.encode(surface, add_special_tokens=False)
        if len(ids) == 1:
            return int(ids[0])
    raise RuntimeError(f"not a single token: {value}")


def _capture_layers(model: Any, input_ids: Any, layers: list[int]) -> dict[int, Any]:
    import torch

    captured: dict[int, Any] = {}
    handles = []
    for layer in layers:
        def hook(module: Any, inputs: object, output: object, layer: int = layer) -> None:
            tensor = output if torch.is_tensor(output) else output[0]
            captured[layer] = tensor.detach()[0].clone()

        handles.append(model.layers[layer].register_forward_hook(hook))
    try:
        with torch.inference_mode():
            model._hf_model(input_ids)
    finally:
        for handle in handles:
            handle.remove()
    return captured


def _output_metrics(
    baseline_log_probs: Any,
    intervened_logits: Any,
    source_answer_id: int,
    target_answer_id: int,
    candidate_ids: list[int],
) -> dict[str, object]:
    log_probs = intervened_logits.float().log_softmax(-1)
    probabilities = log_probs.exp()
    baseline_odds = baseline_log_probs[target_answer_id] - baseline_log_probs[
        source_answer_id
    ]
    intervened_odds = log_probs[target_answer_id] - log_probs[source_answer_id]
    return {
        "baseline_source_log_probability": float(
            baseline_log_probs[source_answer_id].detach().cpu()
        ),
        "baseline_target_log_probability": float(
            baseline_log_probs[target_answer_id].detach().cpu()
        ),
        "intervened_source_log_probability": float(
            log_probs[source_answer_id].detach().cpu()
        ),
        "intervened_target_log_probability": float(
            log_probs[target_answer_id].detach().cpu()
        ),
        "baseline_target_vs_source_logodds": float(baseline_odds.detach().cpu()),
        "intervened_target_vs_source_logodds": float(intervened_odds.detach().cpu()),
        "target_logodds_gain": float((intervened_odds - baseline_odds).detach().cpu()),
        "intervened_top1_token_id": int(log_probs.argmax().detach().cpu()),
        "target_top1": int(log_probs.argmax().detach().cpu()) == target_answer_id,
        "output_kl_nats": float(
            (probabilities * (log_probs - baseline_log_probs)).sum().detach().cpu()
        ),
        "output_entropy_nats": float(
            -(probabilities * log_probs).sum().detach().cpu()
        ),
        "candidate_log_probabilities": {
            str(token_id): float(log_probs[token_id].detach().cpu())
            for token_id in candidate_ids
        },
    }


def _summarize(rows: list[dict[str, object]], keys: list[str]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        values = tuple(
            tuple(row[key]) if isinstance(row[key], list) else row[key]
            for key in keys
        )
        groups[values].append(row)
    summaries: list[dict[str, object]] = []
    for values, part in groups.items():
        eligible = [row for row in part if row["baseline_correct"]]
        denominator = max(len(eligible), 1)
        summaries.append(
            {
                **dict(zip(keys, values, strict=True)),
                "n_trials": len(part),
                "n_eligible": len(eligible),
                "mean_target_logodds_gain": sum(
                    float(row["target_logodds_gain"]) for row in eligible
                )
                / denominator,
                "positive_gain_fraction": sum(
                    float(row["target_logodds_gain"]) > 0 for row in eligible
                )
                / denominator,
                "target_top1_fraction": sum(
                    bool(row["target_top1"]) for row in eligible
                )
                / denominator,
                "mean_output_kl_nats": sum(
                    float(row["output_kl_nats"]) for row in eligible
                )
                / denominator,
                "mean_delta_rms_ratio": sum(
                    float(row["mean_delta_rms_ratio"]) for row in eligible
                )
                / denominator,
                "median_condition_number": sorted(
                    float(row["median_condition_number"]) for row in eligible
                )[len(eligible) // 2]
                if eligible
                else float("nan"),
            }
        )
    return summaries


def _evaluate_country_configs(
    model: Any,
    lens: Any,
    country: dict[str, object],
    configurations: list[dict[str, object]],
) -> list[dict[str, object]]:
    import torch

    from jspace_policy.h0r_diagnostics import (
        directional_coordinate_write,
        donor_coordinate_patch,
        resolve_position_mask,
        symmetric_coordinate_swap,
    )
    from jspace_policy.interventions import MultiLayerResidualIntervention

    arguments = [str(value) for value in country["args"]]
    argument_ids = {arg: _token_id(model.tokenizer, arg) for arg in arguments}
    answer_ids = {
        str(answer): _token_id(model.tokenizer, str(answer))
        for function in country["funcs"]
        for answer in function["answers"].values()
    }
    all_layers = sorted(
        {int(layer) for config in configurations for layer in config["layers"]}
    )
    direction_ids = sorted({*argument_ids.values(), *answer_ids.values()})
    directions = _precompute_directions(model, lens, all_layers, direction_ids)
    candidate_answer_ids = sorted(set(answer_ids.values()))
    rows: list[dict[str, object]] = []

    for function in country["funcs"]:
        answers = {str(key): str(value) for key, value in function["answers"].items()}
        for source in arguments:
            prompt = str(function["template"]).format(arg=source)
            input_ids = model.encode(prompt, max_length=512)
            argument_positions = _argument_positions(model.tokenizer, prompt, source)
            source_answer_id = answer_ids[answers[source]]
            with torch.inference_mode():
                baseline_logits = model._hf_model(input_ids).logits[0, -1].float()
            baseline_log_probs = baseline_logits.log_softmax(-1)
            baseline_top1 = int(baseline_logits.argmax().detach().cpu())

            variants = [
                (target, config)
                for target in arguments
                if target != source
                for config in configurations
            ]
            needs_donor = any(
                config["operation"] == "donor_coordinate_patch"
                for _, config in variants
            )
            donor_states: dict[tuple[str, int], Any] = {}
            if needs_donor:
                for target in arguments:
                    if target == source:
                        continue
                    donor_prompt = str(function["template"]).format(arg=target)
                    donor_ids = model.encode(donor_prompt, max_length=512)
                    captured = _capture_layers(model, donor_ids, all_layers)
                    for layer, value in captured.items():
                        if value.shape[0] == input_ids.shape[1]:
                            donor_states[(target, layer)] = value

            batch_ids = input_ids.repeat(len(variants), 1)
            diagnostics: dict[int, dict[int, dict[str, object]]] = defaultdict(dict)

            def make_transform(
                layer: int,
                current_variants: list[tuple[str, dict[str, object]]],
                current_argument_positions: list[int],
                current_source: str,
                current_source_answer_id: int,
                current_answers: dict[str, str],
                current_donor_states: dict[tuple[str, int], Any],
                current_diagnostics: dict[int, dict[int, dict[str, object]]],
            ):
                def transform(tensor: Any) -> Any:
                    for index, (target, config) in enumerate(current_variants):
                        if layer not in config["layers"]:
                            continue
                        positions = resolve_position_mask(
                            str(config["position_mask"]),
                            sequence_length=int(tensor.shape[1]),
                            argument_positions=current_argument_positions,
                        )
                        before = tensor[index, positions, :].clone()
                        control = str(config.get("control", "semantic"))
                        direction_target = target
                        source_id = argument_ids[current_source]
                        target_id = argument_ids[target]
                        if control == "unrelated_same_category":
                            direction_target = next(
                                item
                                for item in arguments
                                if item not in {current_source, target}
                            )
                            target_id = argument_ids[direction_target]
                        elif control == "direct_answer_direction":
                            source_id = current_source_answer_id
                            target_id = answer_ids[current_answers[target]]
                        source_vector = directions[layer][source_id]
                        target_vector = directions[layer][target_id]
                        if control == "identity_source_source":
                            after = before
                            record = {
                                "basis": {
                                    "cosine": 1.0,
                                    "absolute_cosine": 1.0,
                                    "condition_number": 1.0,
                                },
                                "coordinates": {
                                    "source_pre": [],
                                    "target_pre": [],
                                    "source_post": [],
                                    "target_post": [],
                                    "fp32_exchange_max_abs_error": 0.0,
                                    "hidden_l2": float(
                                        before.float().norm(dim=-1).mean().cpu()
                                    ),
                                    "delta_l2": 0.0,
                                    "delta_rms_ratio": 0.0,
                                },
                            }
                        else:
                            if control == "norm_matched_random_basis":
                                generator = torch.Generator(device="cpu").manual_seed(
                                    SEED + layer * 1009 + index
                                )
                                source_vector = torch.randn(
                                    source_vector.shape, generator=generator
                                ).to(source_vector.device)
                                target_vector = torch.randn(
                                    target_vector.shape, generator=generator
                                ).to(target_vector.device)
                                source_vector = source_vector / source_vector.norm()
                                target_vector = target_vector - (
                                    target_vector @ source_vector
                                ) * source_vector
                                target_vector = target_vector / target_vector.norm()
                            operation = str(config["operation"])
                            if operation == "pseudoinverse_coordinate_swap":
                                after, record = symmetric_coordinate_swap(
                                    before,
                                    source_vector,
                                    target_vector,
                                    alpha=float(config["alpha"]),
                                )
                            elif operation == "source_suppression_target_installation":
                                matrix = torch.stack(
                                    [source_vector.float(), target_vector.float()], dim=1
                                )
                                coordinate = before.float() @ torch.linalg.pinv(matrix).T
                                scale = float(
                                    coordinate.std(dim=0, unbiased=False).mean().clamp_min(0.1)
                                )
                                after, record = directional_coordinate_write(
                                    before,
                                    source_vector,
                                    target_vector,
                                    source_shift=scale,
                                    target_shift=scale,
                                    alpha=float(config["alpha"]),
                                )
                                record["natural_coordinate_scale"] = scale
                            elif operation == "donor_coordinate_patch":
                                donor = current_donor_states.get(
                                    (direction_target, layer)
                                )
                                if donor is None:
                                    raise RuntimeError(
                                        "donor/source sequence-length mismatch"
                                    )
                                after, record = donor_coordinate_patch(
                                    before,
                                    donor[positions, :],
                                    source_vector,
                                    target_vector,
                                    alpha=float(config["alpha"]),
                                )
                            else:
                                raise ValueError(f"unknown operation: {operation}")
                        tensor[index, positions, :] = after
                        current_diagnostics[index][layer] = record
                    return tensor

                return transform

            transforms = {
                layer: make_transform(
                    layer,
                    variants,
                    argument_positions,
                    source,
                    source_answer_id,
                    answers,
                    donor_states,
                    diagnostics,
                )
                for layer in all_layers
            }
            with torch.inference_mode(), MultiLayerResidualIntervention(
                model.layers, transforms
            ):
                intervened_logits = model._hf_model(batch_ids).logits[:, -1].float()

            for index, (target, config) in enumerate(variants):
                layer_records = diagnostics[index]
                coordinate_records = [
                    value["coordinates"] for value in layer_records.values()
                ]
                bases = [value["basis"] for value in layer_records.values()]
                output = _output_metrics(
                    baseline_log_probs,
                    intervened_logits[index],
                    source_answer_id,
                    answer_ids[answers[target]],
                    candidate_answer_ids,
                )
                rows.append(
                    {
                        "trial_id": uuid.uuid4().hex,
                        "scenario_id": hashlib.sha256(
                            f"countries:{function['name']}:{source}".encode()
                        ).hexdigest()[:16],
                        "category": "countries",
                        "function": function["name"],
                        "prompt": prompt,
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "prompt_token_ids": [int(token) for token in input_ids[0].cpu()],
                        "argument_positions": argument_positions,
                        "source_argument": source,
                        "target_argument": target,
                        "source_answer": answers[source],
                        "source_answer_token_id": source_answer_id,
                        "target_answer": answers[target],
                        "target_answer_token_id": answer_ids[answers[target]],
                        "baseline_top1_token_id": baseline_top1,
                        "baseline_correct": baseline_top1 == source_answer_id,
                        "configuration_id": config["configuration_id"],
                        "layers": config["layers"],
                        "position_mask": config["position_mask"],
                        "operation": config["operation"],
                        "control": config.get("control", "semantic"),
                        "alpha": config["alpha"],
                        "layer_diagnostics": {
                            str(layer): value
                            for layer, value in sorted(layer_records.items())
                        },
                        "mean_delta_rms_ratio": sum(
                            float(value["delta_rms_ratio"])
                            for value in coordinate_records
                        )
                        / max(len(coordinate_records), 1),
                        "median_condition_number": sorted(
                            float(value["condition_number"]) for value in bases
                        )[len(bases) // 2],
                        "maximum_absolute_cosine": max(
                            (float(value["absolute_cosine"]) for value in bases),
                            default=1.0,
                        ),
                        **output,
                    }
                )
    return rows


def _reconstruction_runs(
    model: Any,
    lens: Any,
    country: dict[str, object],
    candidate: dict[str, object],
) -> list[dict[str, object]]:
    import torch

    from jspace_policy.h0r_diagnostics import (
        projected_coordinates,
        reconstruction_fraction,
        symmetric_coordinate_swap,
    )

    arguments = [str(value) for value in country["args"]]
    argument_ids = {arg: _token_id(model.tokenizer, arg) for arg in arguments}
    write_layer = int(candidate["layers"][0])
    downstream = [
        int(layer) for layer in lens.source_layers if int(layer) >= write_layer
    ]
    directions = _precompute_directions(
        model, lens, downstream, sorted(argument_ids.values())
    )
    rows: list[dict[str, object]] = []
    for function in country["funcs"]:
        answers = {str(key): str(value) for key, value in function["answers"].items()}
        answer_ids = {
            arg: _token_id(model.tokenizer, answer) for arg, answer in answers.items()
        }
        for source in arguments:
            prompt = str(function["template"]).format(arg=source)
            input_ids = model.encode(prompt, max_length=512)
            clean = _capture_layers(model, input_ids, downstream)
            with torch.inference_mode():
                baseline_logits = model._hf_model(input_ids).logits[0, -1].float()
            for target in arguments:
                if target == source:
                    continue
                captured: dict[int, Any] = {}
                installed: dict[str, object] = {}
                handles = []

                def make_hook(
                    layer: int,
                    current_source: str,
                    current_target: str,
                    current_captured: dict[int, Any],
                    current_installed: dict[str, object],
                ):
                    def hook(module: Any, inputs: object, output: object) -> object:
                        tensor = output if torch.is_tensor(output) else output[0]
                        updated = tensor
                        if layer == write_layer:
                            positions = list(range(tensor.shape[1]))
                            updated = tensor.clone()
                            patched, record = symmetric_coordinate_swap(
                                updated[0, positions, :],
                                directions[layer][argument_ids[current_source]],
                                directions[layer][argument_ids[current_target]],
                                alpha=float(candidate["alpha"]),
                            )
                            updated[0, positions, :] = patched
                            current_installed.update(record)
                        current_captured[layer] = updated.detach()[0].clone()
                        if torch.is_tensor(output):
                            return updated
                        return (updated, *output[1:])

                    return hook

                for layer in downstream:
                    handles.append(
                        model.layers[layer].register_forward_hook(
                            make_hook(layer, source, target, captured, installed)
                        )
                    )
                try:
                    with torch.inference_mode():
                        logits = model._hf_model(input_ids).logits[0, -1].float()
                finally:
                    for handle in handles:
                        handle.remove()
                install_coordinates = installed["coordinates"]
                installed_oriented = sum(install_coordinates["source_post"]) / len(
                    install_coordinates["source_post"]
                ) - sum(install_coordinates["target_post"]) / len(
                    install_coordinates["target_post"]
                )
                for layer in downstream:
                    matrix = torch.stack(
                        [
                            directions[layer][argument_ids[source]].float(),
                            directions[layer][argument_ids[target]].float(),
                        ],
                        dim=1,
                    )
                    clean_coordinate = projected_coordinates(clean[layer].float(), matrix)
                    current_coordinate = projected_coordinates(
                        captured[layer].float(), matrix
                    )
                    clean_oriented = float(
                        (clean_coordinate[:, 0] - clean_coordinate[:, 1]).mean().cpu()
                    )
                    current_oriented = float(
                        (current_coordinate[:, 0] - current_coordinate[:, 1]).mean().cpu()
                    )
                    rows.append(
                        {
                            "trial_id": uuid.uuid4().hex,
                            "scenario_id": hashlib.sha256(
                                f"countries:{function['name']}:{source}".encode()
                            ).hexdigest()[:16],
                            "function": function["name"],
                            "source_argument": source,
                            "target_argument": target,
                            "write_layer": write_layer,
                            "measurement_layer": layer,
                            "clean_oriented_coordinate": clean_oriented,
                            "installed_oriented_coordinate": installed_oriented,
                            "current_oriented_coordinate": current_oriented,
                            "reconstruction_fraction": reconstruction_fraction(
                                current_oriented,
                                clean_oriented,
                                installed_oriented,
                            ),
                            "baseline_correct": int(
                                baseline_logits.argmax().cpu()
                            )
                            == answer_ids[source],
                            "target_top1": int(logits.argmax().cpu())
                            == answer_ids[target],
                        }
                    )
    return rows


def _load_model() -> tuple[Any, Any, dict[str, object]]:
    import torch
    import transformers
    from huggingface_hub import model_info
    from jlens import JacobianLens, from_hf

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
    resolved = model_info(MODEL_ID, revision=MODEL_REVISION).sha
    metadata = {
        "model_id": MODEL_ID,
        "model_revision_requested": MODEL_REVISION,
        "model_revision_resolved": resolved,
        "tokenizer_revision": resolved,
        "lens_repo": LENS_REPO,
        "lens_filename": LENS_FILENAME,
        "lens_revision": LENS_REVISION,
        "lens_code_commit": JLENS_COMMIT,
        "lens_source_layers": [int(layer) for layer in lens.source_layers],
        "seed": SEED,
        "dtype": "bfloat16",
        "gpu_requested": "A100-80GB",
        "gpu_actual": torch.cuda.get_device_name(0),
        "torch_version": str(torch.__version__),
        "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda,
    }
    return model, lens, metadata


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
def run_diagnostic_remote(
    phase: str,
    country: dict[str, object],
    diagnostic: dict[str, object],
    context: dict[str, object],
    git_commit: str,
    dirty_tree: bool,
) -> str:
    import torch

    started = time.perf_counter()
    model, lens, metadata = _load_model()
    artifacts: dict[str, list[dict[str, object]]] = {}
    if phase == "layer":
        configurations = [
            {
                "configuration_id": f"layer_{layer}",
                "layers": [layer],
                "position_mask": diagnostic["layer_sweep"]["position_mask"],
                "operation": diagnostic["layer_sweep"]["operation"],
                "alpha": diagnostic["layer_sweep"]["alpha"],
            }
            for layer in diagnostic["layer_sweep"]["layers"]
        ]
        artifacts["layer_sweep"] = _evaluate_country_configs(
            model, lens, country, configurations
        )
    elif phase == "topology":
        top_layers = [int(layer) for layer in context["top_layers"]]
        position_configs = [
            {
                "configuration_id": f"layer_{layer}__{mask}",
                "layers": [layer],
                "position_mask": mask,
                "operation": "pseudoinverse_coordinate_swap",
                "alpha": 1.0,
            }
            for layer in top_layers
            for mask in diagnostic["position_masks"]
        ]
        position_rows = _evaluate_country_configs(
            model, lens, country, position_configs
        )
        position_summary = _summarize(position_rows, ["configuration_id"])
        winners = sorted(
            position_summary,
            key=lambda row: (
                float(row["mean_target_logodds_gain"]),
                float(row["positive_gain_fraction"]),
            ),
            reverse=True,
        )[: int(diagnostic["top_layer_count"])]
        by_id = {config["configuration_id"]: config for config in position_configs}
        strength_configs = [
            {
                **by_id[str(winner["configuration_id"])],
                "configuration_id": (
                    f"{winner['configuration_id']}__alpha_{float(alpha):g}"
                ),
                "alpha": float(alpha),
            }
            for winner in winners
            for alpha in diagnostic["strengths"]
        ]
        artifacts["position_sweep"] = position_rows
        artifacts["strength_sweep"] = _evaluate_country_configs(
            model, lens, country, strength_configs
        )
    elif phase == "cancellation":
        candidate = context["exploratory_candidate"]
        cumulative_configs = [
            {
                "configuration_id": "layers_" + "_".join(map(str, layers)),
                "layers": layers,
                "position_mask": candidate["position_mask"],
                "operation": "pseudoinverse_coordinate_swap",
                "alpha": candidate["alpha"],
            }
            for layers in diagnostic["cumulative_layer_sets"]
        ]
        operation_configs = [
            {
                "configuration_id": f"operation_{operation}",
                "layers": candidate["layers"],
                "position_mask": candidate["position_mask"],
                "operation": operation,
                "alpha": candidate["alpha"],
            }
            for operation in diagnostic["operations"]
        ]
        control_configs = [
            {
                "configuration_id": f"control_{control}",
                "layers": [20]
                if control == "out_of_window_layer"
                else candidate["layers"],
                "position_mask": candidate["position_mask"],
                "operation": "pseudoinverse_coordinate_swap",
                "control": control,
                "alpha": candidate["alpha"],
            }
            for control in diagnostic["controls"]
        ]
        artifacts["cumulative_layers"] = _evaluate_country_configs(
            model, lens, country, cumulative_configs
        )
        operation_rows = _evaluate_country_configs(
            model, lens, country, operation_configs
        )
        artifacts["coordinate_trajectories"] = operation_rows
        artifacts["controls"] = _evaluate_country_configs(
            model, lens, country, control_configs
        )
        artifacts["reconstruction"] = _reconstruction_runs(
            model, lens, country, candidate
        )
    else:
        raise ValueError(f"unknown diagnostic phase: {phase}")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    cache.commit()
    result = {
        "metadata": {
            "schema_version": 1,
            "run_id": uuid.uuid4().hex,
            "created_at": datetime.now(UTC).isoformat(),
            "git_commit": git_commit,
            "dirty_tree": dirty_tree,
            "phase": phase,
            "diagnostic_config_sha256": hashlib.sha256(
                json.dumps(diagnostic, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "elapsed_seconds": elapsed,
            **metadata,
        },
        "artifacts": artifacts,
    }
    return json.dumps(result, allow_nan=False)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _top_layer_context(path: Path, count: int) -> dict[str, object]:
    rows = _read_jsonl(path)
    summary = _summarize(rows, ["configuration_id", "layers"])
    ordered = sorted(
        summary,
        key=lambda row: (
            float(row["mean_target_logodds_gain"]),
            float(row["positive_gain_fraction"]),
            float(row["target_top1_fraction"]),
        ),
        reverse=True,
    )
    return {"top_layers": [int(row["layers"][0]) for row in ordered[:count]]}


def _candidate_context(path: Path) -> dict[str, object]:
    rows = _read_jsonl(path)
    summary = _summarize(
        rows,
        ["configuration_id", "layers", "position_mask", "operation", "alpha"],
    )
    valid = [
        row
        for row in summary
        if float(row["median_condition_number"]) <= 20
        and float(row["mean_output_kl_nats"]) <= 1
        and float(row["mean_delta_rms_ratio"]) <= 0.25
    ]
    if not valid:
        raise RuntimeError("no strength-sweep configuration is inside validity limits")
    winner = max(
        valid,
        key=lambda row: (
            float(row["mean_target_logodds_gain"]),
            float(row["positive_gain_fraction"]),
            float(row["target_top1_fraction"]),
        ),
    )
    return {
        "exploratory_candidate": {
            "layers": winner["layers"],
            "position_mask": winner["position_mask"],
            "operation": winner["operation"],
            "alpha": winner["alpha"],
        }
    }


@app.local_entrypoint()
def diagnostic(phase: str = "layer") -> None:
    """Run one preregistered H0R-B diagnostic GPU phase and download raw rows."""
    from jspace_policy.budget import admit_run, append_ledger, estimate_cost

    if phase not in {"layer", "topology", "cancellation"}:
        raise ValueError("phase must be layer, topology, or cancellation")
    diagnostic_config = json.loads(Path("configs/v2/h0r_diagnostic.json").read_text())
    flexible = json.loads(
        Path("configs/v2/flexible_generalization_smoke.json").read_text()
    )
    country = next(
        item for item in flexible["data"]["categories"] if item["name"] == "countries"
    )
    raw = Path("results/v2_h0r_diagnostic/raw")
    context: dict[str, object] = {}
    if phase == "topology":
        context = _top_layer_context(
            raw / "layer_sweep.jsonl", int(diagnostic_config["top_layer_count"])
        )
    elif phase == "cancellation":
        context = _candidate_context(raw / "strength_sweep.jsonl")
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(["git", "status", "--short"], text=True)
    dirty_tree = bool(
        "\n".join(line for line in status.splitlines() if ".prismor/" not in line)
    )
    estimate = estimate_cost("A100-80GB", 3600, memory_gib=32.0)
    ledger = Path("artifacts/raw/cost_ledger.jsonl")
    admit_run(ledger, estimate)
    result = json.loads(
        run_diagnostic_remote.remote(
            phase,
            country,
            diagnostic_config,
            context,
            git_commit,
            dirty_tree,
        )
    )
    raw.mkdir(parents=True, exist_ok=True)
    metadata = result["metadata"]
    for name, rows in result["artifacts"].items():
        with (raw / f"{name}.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(metadata | row, sort_keys=True) + "\n")
    manifest = Path("results/v2_h0r_diagnostic/run_manifest.json")
    existing = json.loads(manifest.read_text()) if manifest.exists() else {"runs": []}
    existing["runs"].append(metadata)
    manifest.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
    measured = estimate_cost(
        "A100-80GB", float(metadata["elapsed_seconds"]), memory_gib=32.0
    )
    append_ledger(
        ledger,
        measured,
        run_id=str(metadata["run_id"]),
        stage=f"v2-h0r-diagnostic-{phase}",
    )
    print(
        json.dumps(
            {
                "phase": phase,
                "run_id": metadata["run_id"],
                "elapsed_seconds": metadata["elapsed_seconds"],
                "recorded_cost_usd": measured.buffered_usd,
                "rows": {
                    name: len(rows) for name, rows in result["artifacts"].items()
                },
                "context": context,
            },
            indent=2,
        )
    )
