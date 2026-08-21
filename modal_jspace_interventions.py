"""Modal runners for V3-JI1 J-space intervention replication."""

from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

MODEL_ID = "Qwen/Qwen3.6-27B"
MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
JLENS_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
LENS_FILENAME = "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt"
SEED = 1729
RESULTS = Path("results/v3_jspace_interventions")
CONFIG_DIR = Path("configs/v3/jspace_interventions")
SCOPED_FREEZE_PATHS = [
    "modal_jspace_interventions.py",
    "src/jspace_policy/jspace_interventions.py",
    "tests/test_jspace_interventions.py",
    "scripts/freeze_jspace_intervention_dataset.py",
    "scripts/analyze_jspace_interventions.py",
    "configs/v3/jspace_interventions",
    "docs/v3",
]

app = modal.App("jspace-v3-ji1")
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


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_model() -> tuple[Any, Any, dict[str, Any]]:
    import torch
    import transformers
    from huggingface_hub import model_info
    from jlens import JacobianLens, from_hf

    torch.manual_seed(SEED)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
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
    resolved = model_info(MODEL_ID, revision=MODEL_REVISION).sha
    if lens.d_model != model.d_model:
        raise RuntimeError(f"lens width {lens.d_model} != model width {model.d_model}")
    metadata = {
        "model_name": MODEL_ID,
        "model_revision": resolved,
        "tokenizer_revision": resolved,
        "lens_repository": LENS_REPO,
        "lens_revision": LENS_REVISION,
        "lens_filename": LENS_FILENAME,
        "lens_reference_commit": JLENS_COMMIT,
        "lens_source_layers": [int(layer) for layer in lens.source_layers],
        "torch_version": str(torch.__version__),
        "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_type": torch.cuda.get_device_name(0),
        "dtype": "bfloat16",
        "seed": SEED,
    }
    return model, lens, metadata


def _input_ids(tokenizer: Any, prompt: str) -> Any:
    import torch

    return torch.tensor(
        [tokenizer.encode(prompt, add_special_tokens=False)],
        device="cuda",
        dtype=torch.long,
    )


def _candidate_cell(
    tokenizer: Any, category: str, function: dict[str, Any], argument: str
) -> dict[str, Any]:
    prompt = str(function["template"]).format(arg=argument)
    answer = str(function["answers"][argument])
    encoded = tokenizer(
        prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    prompt_ids = [int(value) for value in encoded["input_ids"]]
    start = prompt.find(argument)
    end = start + len(argument)
    argument_positions = [
        index
        for index, (left, right) in enumerate(encoded["offset_mapping"])
        if int(left) < end and int(right) > start
    ]
    contextual_argument_ids = [prompt_ids[index] for index in argument_positions]
    canonical_argument_ids = tokenizer.encode(f" {argument}", add_special_tokens=False)
    continuation = f" {answer}"
    full_ids = tokenizer.encode(prompt + continuation, add_special_tokens=False)
    added = full_ids[len(prompt_ids) :] if full_ids[: len(prompt_ids)] == prompt_ids else []
    tokenization_eligible = (
        start >= 0
        and len(argument_positions) == 1
        and len(canonical_argument_ids) == 1
        and len(added) == 1
    )
    return {
        "category": category,
        "function": function["name"],
        "argument": argument,
        "answer": answer,
        "prompt": prompt,
        "prompt_token_ids": prompt_ids,
        "argument_positions": argument_positions,
        "contextual_argument_token_ids": contextual_argument_ids,
        "canonical_argument_token_id": int(canonical_argument_ids[0])
        if len(canonical_argument_ids) == 1
        else None,
        "answer_token_id": int(added[0]) if len(added) == 1 else None,
        "decoded_answer_token": tokenizer.decode(added) if len(added) == 1 else None,
        "tokenization_eligible": tokenization_eligible,
    }


@app.function(
    image=image,
    volumes={"/cache": cache},
    cpu=4,
    memory=32768,
    gpu="A100-80GB",
    timeout=1800,
    retries=0,
)
def behavior_only_remote(candidates: dict[str, Any], context: dict[str, Any]) -> str:
    import torch

    started = time.perf_counter()
    model, _, metadata = _load_model()
    rows: list[dict[str, Any]] = []
    for category in candidates["categories"]:
        for function in category["functions"]:
            for argument in category["arguments"]:
                row = _candidate_cell(
                    model.tokenizer, str(category["name"]), function, str(argument)
                )
                if row["tokenization_eligible"]:
                    ids = _input_ids(model.tokenizer, str(row["prompt"]))
                    with torch.inference_mode():
                        logits = model._hf_model(ids).logits[0, -1].float()
                    top1 = int(logits.argmax().detach().cpu())
                    row["clean_top1_token_id"] = top1
                    row["clean_top1_decoded"] = model.tokenizer.decode([top1])
                    row["clean_correct"] = top1 == int(row["answer_token_id"])
                    top = logits.log_softmax(-1).topk(5)
                    row["clean_top5"] = [
                        {
                            "token_id": int(token_id),
                            "token": model.tokenizer.decode([int(token_id)]),
                            "log_probability": float(log_probability),
                        }
                        for token_id, log_probability in zip(
                            top.indices.detach().cpu().tolist(),
                            top.values.detach().cpu().tolist(),
                            strict=True,
                        )
                    ]
                else:
                    row["clean_top1_token_id"] = None
                    row["clean_top1_decoded"] = None
                    row["clean_correct"] = False
                    row["clean_top5"] = []
                rows.append(row)
    cache.commit()
    artifact = {
        "schema_version": 1,
        "study_id": "V3-JI1",
        "run_id": uuid.uuid4().hex,
        "created_at": datetime.now(UTC).isoformat(),
        "intervention_opened": False,
        "candidate_sha256": _canonical_sha256(candidates),
        "elapsed_seconds": time.perf_counter() - started,
        "context": context,
        "metadata": metadata,
        "rows": rows,
    }
    return json.dumps(artifact, allow_nan=False)


def _directions(
    model: Any,
    lens: Any,
    layers: list[int],
    token_ids: list[int],
    *,
    unit_l2: bool,
) -> dict[int, dict[int, Any]]:
    weights = model._hf_model.lm_head.weight.detach()[token_ids].float()
    result: dict[int, dict[int, Any]] = {}
    for layer in layers:
        jacobian = lens.jacobians[layer].to(weights.device).float()
        vectors = weights @ jacobian
        if unit_l2:
            vectors = vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        result[layer] = {
            token_id: vectors[index].detach() for index, token_id in enumerate(token_ids)
        }
        del jacobian
    return result


def _capture_clean(model: Any, input_ids: Any, layers: list[int]) -> tuple[Any, dict[int, Any]]:
    import torch

    captured: dict[int, Any] = {}
    handles = []
    for layer in layers:

        def hook(module: Any, inputs: object, output: object, layer: int = layer) -> None:
            tensor = output if torch.is_tensor(output) else output[0]
            captured[layer] = tensor.detach()[0].float().clone()

        handles.append(model.layers[layer].register_forward_hook(hook))
    try:
        with torch.inference_mode():
            logits = model._hf_model(input_ids).logits[0, -1].float()
    finally:
        for handle in handles:
            handle.remove()
    return logits, captured


def _rank(logits: Any, token_id: int) -> int:
    return int((logits > logits[token_id]).sum().detach().cpu()) + 1


def _output_metrics(
    clean_logits: Any, patched_logits: Any, source_id: int, target_id: int, tokenizer: Any
) -> dict[str, Any]:
    clean_logp = clean_logits.float().log_softmax(-1)
    patched_logp = patched_logits.float().log_softmax(-1)
    clean_p = clean_logp.exp()
    clean_margin = clean_logp[target_id] - clean_logp[source_id]
    patched_margin = patched_logp[target_id] - patched_logp[source_id]
    top = patched_logp.topk(10)
    return {
        "baseline_source_rank": _rank(clean_logits, source_id),
        "baseline_target_rank": _rank(clean_logits, target_id),
        "patched_source_rank": _rank(patched_logits, source_id),
        "patched_target_rank": _rank(patched_logits, target_id),
        "baseline_source_logprob": float(clean_logp[source_id].detach().cpu()),
        "baseline_target_logprob": float(clean_logp[target_id].detach().cpu()),
        "patched_source_logprob": float(patched_logp[source_id].detach().cpu()),
        "patched_target_logprob": float(patched_logp[target_id].detach().cpu()),
        "target_minus_source_logodds_clean": float(clean_margin.detach().cpu()),
        "target_minus_source_logodds_patched": float(patched_margin.detach().cpu()),
        "delta_target_minus_source_logodds": float(
            (patched_margin - clean_margin).detach().cpu()
        ),
        "target_top1_clean": int(clean_logits.argmax().detach().cpu()) == target_id,
        "target_top1_patched": int(patched_logits.argmax().detach().cpu()) == target_id,
        "source_top1_clean": int(clean_logits.argmax().detach().cpu()) == source_id,
        "source_top1_patched": int(patched_logits.argmax().detach().cpu()) == source_id,
        "output_kl_clean_to_patched": float(
            (clean_p * (clean_logp - patched_logp)).sum().detach().cpu()
        ),
        "output_entropy_clean": float(-(clean_p * clean_logp).sum().detach().cpu()),
        "output_entropy_patched": float(
            -(patched_logp.exp() * patched_logp).sum().detach().cpu()
        ),
        "patched_top10": [
            {
                "token_id": int(token_id),
                "token": tokenizer.decode([int(token_id)]),
                "log_probability": float(log_probability),
            }
            for token_id, log_probability in zip(
                top.indices.detach().cpu().tolist(),
                top.values.detach().cpu().tolist(),
                strict=True,
            )
        ],
    }


def _normalise_categories(dataset: dict[str, Any], tokenizer: Any) -> list[dict[str, Any]]:
    categories: list[dict[str, Any]] = []
    source_categories = dataset.get("data", {}).get("categories", dataset.get("categories", []))
    for category in source_categories:
        arguments = [str(value) for value in category.get("args", category.get("arguments"))]
        functions = category.get("funcs", category.get("functions"))
        normalized_functions = []
        for function in functions:
            normalized_functions.append(
                {
                    "name": str(function["name"]),
                    "template": str(function["template"]),
                    "answers": {
                        str(key): str(value) for key, value in function["answers"].items()
                    },
                    "cells": function.get("cells", {}),
                }
            )
        categories.append(
            {
                "name": str(category.get("name", category.get("category"))),
                "arguments": arguments,
                "functions": normalized_functions,
            }
        )
    return categories


def _scenario_metadata(
    tokenizer: Any, category: dict[str, Any], function: dict[str, Any], source: str
) -> dict[str, Any]:
    prompt = function["template"].format(arg=source)
    cell = function.get("cells", {}).get(source)
    if cell:
        return {
            "prompt": prompt,
            "prompt_token_ids": [int(value) for value in cell["prompt_token_ids"]],
            "argument_positions": [int(value) for value in cell["argument_positions"]],
            "source_concept_token_id": int(cell["canonical_argument_token_id"]),
            "source_answer_token_id": int(cell["answer_token_id"]),
        }
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    start = prompt.find(source)
    end = start + len(source)
    positions = [
        index
        for index, (left, right) in enumerate(encoded["offset_mapping"])
        if int(left) < end and int(right) > start
    ]
    source_id = _single_token_id(tokenizer, source)
    answer_id = _single_token_id(tokenizer, function["answers"][source])
    if not positions:
        raise RuntimeError(
            "burned cell is not single-token compatible: "
            f"{category['name']} {function['name']} {source}"
        )
    return {
        "prompt": prompt,
        "prompt_token_ids": [int(value) for value in encoded["input_ids"]],
        "argument_positions": positions,
        "source_concept_token_id": source_id,
        "source_answer_token_id": answer_id,
    }


def _single_token_id(tokenizer: Any, value: str) -> int:
    """Match H0/H0R: prefer the leading-space form, then standalone."""
    for surface in (f" {value}", value):
        token_ids = tokenizer.encode(surface, add_special_tokens=False)
        if len(token_ids) == 1:
            return int(token_ids[0])
    raise RuntimeError(f"not a suitable single token in either surface form: {value}")


def _target_ids(tokenizer: Any, function: dict[str, Any], target: str) -> tuple[int, int]:
    cell = function.get("cells", {}).get(target)
    if cell:
        return int(cell["canonical_argument_token_id"]), int(cell["answer_token_id"])
    return _single_token_id(tokenizer, target), _single_token_id(
        tokenizer, function["answers"][target]
    )


def _conditions(experiment: dict[str, Any], *, phase: str) -> list[dict[str, Any]]:
    primary = experiment["primary"]
    result = []
    for alpha in primary["alphas"]:
        for control in ["semantic", *primary["controls"]]:
            result.append(
                {
                    "condition_id": f"native_raw_a{float(alpha):g}_{control}",
                    "vector_semantics": "native_raw",
                    "layers": primary["layers"],
                    "position_mask": primary["position_mask"],
                    "alpha": float(alpha),
                    "control_kind": control,
                }
            )
    if phase == "burned":
        for diagnostic in experiment["burned_diagnostics"]:
            result.append(
                {
                    "condition_id": diagnostic["id"],
                    "vector_semantics": diagnostic["vector_semantics"],
                    "layers": diagnostic["layers"],
                    "position_mask": diagnostic["position_mask"],
                    "alpha": float(diagnostic["alpha"]),
                    "control_kind": "semantic",
                }
            )
    return result


def _loading(
    clean: dict[int, Any],
    directions: dict[int, dict[int, Any]],
    layers: list[int],
    source_id: int,
    argument_positions: list[int],
) -> tuple[float, float]:
    import torch

    paper_values = []
    final_values = []
    for layer in layers:
        direction = directions[layer][source_id].float()
        values = clean[layer]
        for position in [*argument_positions, values.shape[0] - 1]:
            paper_values.append(
                float(
                    torch.nn.functional.cosine_similarity(values[position], direction, dim=0)
                    .detach()
                    .cpu()
                )
            )
        final_values.append(
            float(
                torch.nn.functional.cosine_similarity(values[-1], direction, dim=0)
                .detach()
                .cpu()
            )
        )
    return sum(paper_values) / len(paper_values), sum(final_values) / len(final_values)


def _evaluate(
    model: Any, lens: Any, dataset: dict[str, Any], experiment: dict[str, Any], *, phase: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import torch

    from jspace_policy.interventions import MultiLayerResidualIntervention
    from jspace_policy.jspace_interventions import (
        balanced_unrelated_target,
        coordinate_swap_fp32,
        matched_random_swap_fp32,
        resolve_position_mask,
    )

    categories = _normalise_categories(dataset, model.tokenizer)
    conditions = _conditions(experiment, phase=phase)
    all_layers = sorted(
        {int(layer) for condition in conditions for layer in condition["layers"]}
    )
    token_ids: set[int] = set()
    for category in categories:
        for argument in category["arguments"]:
            token_ids.add(_single_token_id(model.tokenizer, argument))
    raw_directions = _directions(model, lens, all_layers, sorted(token_ids), unit_l2=False)
    unit_directions = _directions(model, lens, all_layers, sorted(token_ids), unit_l2=True)
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []

    for category in categories:
        arguments = category["arguments"]
        for function in category["functions"]:
            for source in arguments:
                try:
                    scenario = _scenario_metadata(model.tokenizer, category, function, source)
                except RuntimeError as exc:
                    exclusions.append(
                        {
                            "category": category["name"],
                            "function": function["name"],
                            "source_argument": source,
                            "target_argument": None,
                            "reason": str(exc),
                        }
                    )
                    continue
                input_ids = torch.tensor(
                    [scenario["prompt_token_ids"]], device="cuda", dtype=torch.long
                )
                clean_logits, clean_states = _capture_clean(model, input_ids, all_layers)
                source_id = int(scenario["source_concept_token_id"])
                source_answer_id = int(scenario["source_answer_token_id"])
                paper_loading, final_loading = _loading(
                    clean_states,
                    raw_directions,
                    [int(layer) for layer in experiment["primary"]["layers"]],
                    source_id,
                    scenario["argument_positions"],
                )
                scenario_id = hashlib.sha256(
                    f"{category['name']}:{function['name']}:{source}".encode()
                ).hexdigest()[:16]
                for target in arguments:
                    if target == source:
                        continue
                    try:
                        target_id, target_answer_id = _target_ids(
                            model.tokenizer, function, target
                        )
                    except RuntimeError as exc:
                        exclusions.append(
                            {
                                "category": category["name"],
                                "function": function["name"],
                                "source_argument": source,
                                "target_argument": target,
                                "reason": str(exc),
                            }
                        )
                        continue
                    unrelated = balanced_unrelated_target(arguments, source, target)
                    unrelated_id = _single_token_id(model.tokenizer, unrelated)
                    for condition in conditions:
                        diagnostics: list[dict[str, Any]] = []
                        if condition["control_kind"] == "identity":
                            patched_logits = clean_logits
                        else:
                            selected_directions = (
                                raw_directions
                                if condition["vector_semantics"] == "native_raw"
                                else unit_directions
                            )
                            transforms: dict[int, Any] = {}
                            for layer_value in condition["layers"]:
                                layer = int(layer_value)
                                positions = resolve_position_mask(
                                    str(condition["position_mask"]),
                                    sequence_length=int(input_ids.shape[1]),
                                    argument_positions=scenario["argument_positions"],
                                )
                                current_source = selected_directions[layer][source_id]
                                semantic_target = selected_directions[layer][target_id]
                                control_kind = str(condition["control_kind"])
                                alpha = float(condition["alpha"])
                                patch_target = (
                                    selected_directions[layer][unrelated_id]
                                    if condition["control_kind"] == "unrelated_semantic"
                                    else semantic_target
                                )
                                stable_seed = int(
                                    hashlib.sha256(
                                        f"{SEED}:{scenario_id}:{target}:{condition['condition_id']}:{layer}".encode()
                                    ).hexdigest()[:8],
                                    16,
                                )

                                def transform(
                                    hidden: Any,
                                    *,
                                    positions: list[int] = positions,
                                    layer: int = layer,
                                    current_source: Any = current_source,
                                    patch_target: Any = patch_target,
                                    semantic_target: Any = semantic_target,
                                    stable_seed: int = stable_seed,
                                    control_kind: str = control_kind,
                                    alpha: float = alpha,
                                    diagnostic_records: list[dict[str, Any]] = diagnostics,
                                ) -> Any:
                                    updated = hidden.clone()
                                    selected = updated[:, positions, :]
                                    if control_kind == "random_delta_matched":
                                        patched, record = matched_random_swap_fp32(
                                            selected,
                                            current_source,
                                            semantic_target,
                                            alpha=alpha,
                                            seed=stable_seed,
                                        )
                                    else:
                                        patched, swap_record = coordinate_swap_fp32(
                                            selected,
                                            current_source,
                                            patch_target,
                                            alpha=alpha,
                                        )
                                        record = {"semantic": swap_record.__dict__}
                                    updated[:, positions, :] = patched
                                    diagnostic_records.append(
                                        {"layer": layer, "positions": positions, **record}
                                    )
                                    return updated

                                transforms[layer] = transform
                            with MultiLayerResidualIntervention(
                                model.layers, transforms, positions=None
                            ):
                                with torch.inference_mode():
                                    patched_logits = (
                                        model._hf_model(input_ids).logits[0, -1].float()
                                    )

                        semantic_records = [
                            record.get("semantic", {}) for record in diagnostics
                        ]
                        geometry = [
                            record
                            for record in semantic_records
                            if record.get("geometry_eligible")
                        ]
                        delta_means = [
                            float(record.get("semantic", {}).get("delta_rms_ratio_mean", 0.0))
                            for record in diagnostics
                        ]
                        delta_maxima = [
                            float(record.get("semantic", {}).get("delta_rms_ratio_max", 0.0))
                            for record in diagnostics
                        ]
                        if not delta_means:
                            delta_means = [0.0]
                            delta_maxima = [0.0]
                        row = {
                            "study_id": "V3-JI1",
                            "phase": phase,
                            "condition_id": condition["condition_id"],
                            "scenario_id": scenario_id,
                            "cluster_id": scenario_id,
                            "category": category["name"],
                            "function": function["name"],
                            "source_argument": source,
                            "target_argument": target,
                            "unrelated_argument": unrelated,
                            "prompt": scenario["prompt"],
                            "prompt_token_ids": scenario["prompt_token_ids"],
                            "source_concept_token_id": source_id,
                            "target_concept_token_id": target_id,
                            "source_answer": function["answers"][source],
                            "target_answer": function["answers"][target],
                            "source_answer_token_id": source_answer_id,
                            "target_answer_token_id": target_answer_id,
                            "vector_semantics": condition["vector_semantics"],
                            "layers": condition["layers"],
                            "position_mask": condition["position_mask"],
                            "alpha": condition["alpha"],
                            "control_kind": condition["control_kind"],
                            "source_workspace_loading_paper": paper_loading,
                            "source_workspace_loading_final": final_loading,
                            "mean_delta_rms_ratio": sum(delta_means) / len(delta_means),
                            "max_delta_rms_ratio": max(delta_maxima),
                            "median_basis_cosine": sorted(
                                float(record["cosine"]) for record in geometry
                            )[len(geometry) // 2]
                            if geometry
                            else None,
                            "max_basis_cosine": max(
                                abs(float(record["cosine"])) for record in geometry
                            )
                            if geometry
                            else None,
                            "median_condition_number": sorted(
                                float(record["condition_number"]) for record in geometry
                            )[len(geometry) // 2]
                            if geometry
                            else None,
                            "max_coordinate_target_error": max(
                                float(record.get("coordinate_target_error", 0.0))
                                for record in semantic_records
                            )
                            if semantic_records
                            else 0.0,
                            "eligible": int(clean_logits.argmax().detach().cpu())
                            == source_answer_id,
                            "exclusion_reason": None
                            if int(clean_logits.argmax().detach().cpu()) == source_answer_id
                            else "clean_source_not_top1",
                            **_output_metrics(
                                clean_logits,
                                patched_logits,
                                source_answer_id,
                                target_answer_id,
                                model.tokenizer,
                            ),
                        }
                        rows.append(row)
    return rows, exclusions


def _phase_a(model: Any, lens: Any) -> dict[str, Any]:
    import torch

    from jspace_policy.interventions import MultiLayerResidualIntervention
    from jspace_policy.jspace_interventions import coordinate_swap_fp32

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    maximum_error = 0.0
    for alpha in [0.0, 0.5, 1.0, 2.0]:
        for _ in range(16):
            hidden = torch.randn((7, 64), generator=generator, dtype=torch.float32).cuda()
            source = torch.randn((64,), generator=generator, dtype=torch.float32).cuda()
            target = torch.randn((64,), generator=generator, dtype=torch.float32).cuda()
            _, diagnostics = coordinate_swap_fp32(hidden, source, target, alpha=alpha)
            maximum_error = max(maximum_error, diagnostics.coordinate_target_error)
    identity_hidden = torch.randn((3, 64), generator=generator, dtype=torch.float32).cuda()
    identity_source = torch.randn((64,), generator=generator, dtype=torch.float32).cuda()
    identity, identity_diagnostics = coordinate_swap_fp32(
        identity_hidden, identity_source, identity_source, alpha=2.0
    )
    prompt = "The capital of Germany is the city of"
    input_ids = _input_ids(model.tokenizer, prompt)
    with torch.inference_mode():
        clean = model._hf_model(input_ids).logits[0, -1].float()
    with MultiLayerResidualIntervention(
        model.layers, {36: lambda value: value.clone()}, positions=None
    ):
        with torch.inference_mode():
            hooked = model._hf_model(input_ids).logits[0, -1].float()
    clean_parity_error = float((clean - hooked).abs().max().detach().cpu())
    result = {
        "coordinate_max_abs_error": maximum_error,
        "coordinate_gate_pass": maximum_error <= 2e-5,
        "identity_exact": bool(torch.equal(identity, identity_hidden)),
        "identity_geometry_eligible": identity_diagnostics.geometry_eligible,
        "clean_hook_logit_max_abs_error": clean_parity_error,
        "clean_hook_parity_pass": clean_parity_error == 0.0,
        "layer_36_in_lens": 36 in [int(layer) for layer in lens.source_layers],
        "layer_43_in_lens": 43 in [int(layer) for layer in lens.source_layers],
    }
    result["all_pass"] = all(
        [
            result["coordinate_gate_pass"],
            result["identity_exact"],
            not result["identity_geometry_eligible"],
            result["clean_hook_parity_pass"],
            result["layer_36_in_lens"],
            result["layer_43_in_lens"],
        ]
    )
    return result


@app.function(
    image=image,
    volumes={"/cache": cache},
    cpu=4,
    memory=32768,
    gpu="A100-80GB",
    timeout=7200,
    retries=0,
)
def study_remote(
    phase: str, dataset: dict[str, Any], experiment: dict[str, Any], context: dict[str, Any]
) -> str:
    import torch

    started = time.perf_counter()
    model, lens, metadata = _load_model()
    phase_a = _phase_a(model, lens)
    if not phase_a["all_pass"]:
        raise RuntimeError(f"Phase A failed: {phase_a}")
    rows, exclusions = _evaluate(model, lens, dataset, experiment, phase=phase)
    torch.cuda.synchronize()
    cache.commit()
    artifact = {
        "schema_version": 1,
        "study_id": "V3-JI1",
        "run_id": uuid.uuid4().hex,
        "created_at": datetime.now(UTC).isoformat(),
        "phase": phase,
        "phase_a": phase_a,
        "dataset_sha256": _canonical_sha256(dataset),
        "experiment_sha256": _canonical_sha256(experiment),
        "elapsed_seconds": time.perf_counter() - started,
        "context": context,
        "metadata": metadata,
        "exclusions": exclusions,
        "rows": rows,
    }
    return json.dumps(artifact, allow_nan=False)


def _git_context(*, require_scoped_clean: bool) -> dict[str, Any]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    whole_status = subprocess.check_output(["git", "status", "--short"], text=True)
    scoped_status = subprocess.check_output(
        ["git", "status", "--short", "--", *SCOPED_FREEZE_PATHS], text=True
    )
    if require_scoped_clean and scoped_status.strip():
        raise RuntimeError(
            "V3 study paths are not committed; refusing sealed run:\n" + scoped_status
        )
    return {
        "git_commit": commit,
        "whole_tree_dirty": bool(whole_status.strip()),
        "study_paths_clean": not bool(scoped_status.strip()),
        "unrelated_dirty_paths_recorded": [
            line for line in whole_status.splitlines() if line not in scoped_status.splitlines()
        ],
    }


def _write_json(path: Path, payload: str) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload + "\n")


def _write_gzip(path: Path, payload: str) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            compressed.write((payload + "\n").encode())


@app.local_entrypoint()
def behavior_only() -> None:
    candidates = json.loads((CONFIG_DIR / "fresh_candidates.json").read_text())
    context = _git_context(require_scoped_clean=True)
    payload = behavior_only_remote.remote(candidates, context)
    output = RESULTS / "raw/fresh_baseline_candidates.json"
    _write_json(output, payload)
    parsed = json.loads(payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "run_id": parsed["run_id"],
                "rows": len(parsed["rows"]),
                "tokenization_eligible": sum(
                    row["tokenization_eligible"] for row in parsed["rows"]
                ),
                "clean_correct": sum(row["clean_correct"] for row in parsed["rows"]),
                "intervention_opened": parsed["intervention_opened"],
            },
            indent=2,
        )
    )


@app.local_entrypoint()
def run_phase(phase: str) -> None:
    if phase not in {"burned", "fresh"}:
        raise ValueError("phase must be burned or fresh")
    experiment = json.loads((CONFIG_DIR / "experiment.json").read_text())
    if phase == "burned":
        dataset = json.loads(Path("configs/v2/flexible_generalization_smoke.json").read_text())
        output = RESULTS / "raw/burned_replication.json.gz"
        require_scoped_clean = True
    else:
        dataset = json.loads((CONFIG_DIR / "fresh_frozen.json").read_text())
        if dataset["status"] != "behavior_only_frozen_interventions_unopened":
            raise RuntimeError("fresh dataset status is not sealed")
        output = RESULTS / "raw/fresh_interventions.json.gz"
        require_scoped_clean = True
    context = _git_context(require_scoped_clean=require_scoped_clean)
    payload = study_remote.remote(phase, dataset, experiment, context)
    _write_gzip(output, payload)
    parsed = json.loads(payload)
    manifest = {key: value for key, value in parsed.items() if key != "rows"}
    manifest["raw_output_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest_path = RESULTS / "manifests" / f"phase_{'b' if phase == 'burned' else 'd'}.json"
    _write_json(manifest_path, json.dumps(manifest, indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                "phase": phase,
                "output": str(output),
                "manifest": str(manifest_path),
                "run_id": parsed["run_id"],
                "rows": len(parsed["rows"]),
                "phase_a_pass": parsed["phase_a"]["all_pass"],
                "elapsed_seconds": parsed["elapsed_seconds"],
                "outcomes_not_summarized_by_runner": True,
            },
            indent=2,
        )
    )
