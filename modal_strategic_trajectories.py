"""Gated execution for the V2-E2 MVP Strategic J-Space Trajectories study.

Execution order:

    modal run modal_strategic_trajectories.py::freeze_dataset
    modal run modal_strategic_trajectories.py::behavior
    modal run modal_strategic_trajectories.py::mechanistic

The mechanistic entrypoint refuses to run unless all conjunctive behavioral
gates passed.  No causal intervention or Patchscope entrypoint is exposed.
"""

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

from jspace_policy.budget import admit_run, append_ledger, estimate_cost

CONFIG_PATH = Path("configs/v2/strategic_trajectories/experiment.json")
SOURCE_DATASET_PATH = Path("configs/v2/strategic_trajectories/dataset_source.json")
TOKENIZED_DATASET_PATH = Path("configs/v2/strategic_trajectories/dataset.json")
RESULT_ROOT = Path("results/v2_strategic_trajectories")

MODEL_ID = "Qwen/Qwen3.6-27B"
MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
JLENS_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
LENS_FILENAME = (
    "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt"
)

app = modal.App("jspace-v2-e2-strategic-trajectories")
cache = modal.Volume.from_name("jspace-hf-cache", create_if_missing=True)
artifacts = modal.Volume.from_name(
    "jspace-v2-e2-trajectory-artifacts", create_if_missing=True
)
image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .uv_pip_install(
        "numpy>=2.0",
        "torch>=2.8",
        "transformers>=5.5",
        "huggingface_hub>=0.34",
        f"git+https://github.com/anthropics/jacobian-lens.git@{JLENS_COMMIT}",
    )
    .env({"HF_HOME": "/cache/huggingface", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir("src/jspace_policy", remote_path="/root/jspace_policy")
)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _hash_valid(payload: dict[str, Any]) -> bool:
    claimed = payload.get("content_sha256")
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    return claimed == _canonical_sha256(body)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_config_pins(config: dict[str, Any]) -> None:
    expected = {
        "model.id": (config["model"]["id"], MODEL_ID),
        "model.revision": (config["model"]["revision"], MODEL_REVISION),
        "model.tokenizer_revision": (
            config["model"]["tokenizer_revision"],
            MODEL_REVISION,
        ),
        "lens.repo": (config["lens"]["repo"], LENS_REPO),
        "lens.revision": (config["lens"]["revision"], LENS_REVISION),
        "lens.filename": (config["lens"]["filename"], LENS_FILENAME),
        "lens.code_commit": (config["lens"]["code_commit"], JLENS_COMMIT),
        "trajectory.layers": (config["trajectory"]["layers"], [34, 42, 46, 54, 60]),
    }
    mismatches = [name for name, (actual, frozen) in expected.items() if actual != frozen]
    if mismatches:
        raise RuntimeError(f"configuration changed frozen pins: {mismatches}")
    if config["execution"]["causal_interventions_authorized"] is not False:
        raise RuntimeError("V2-E2 MVP cannot authorize causal interventions")


def _write_new(path: Path, text: str) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite prospective artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _load_model(*, with_lens: bool) -> tuple[Any, Any | None, dict[str, Any]]:
    import torch
    import transformers
    from huggingface_hub import model_info
    from jlens import JacobianLens, from_hf

    torch.manual_seed(82026)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).cuda()
    hf_model.eval()
    model = from_hf(hf_model, tokenizer)
    if int(model.n_layers) != 64:
        raise RuntimeError(f"pinned model layer count changed: {model.n_layers}")
    lens = None
    if with_lens:
        lens = JacobianLens.from_pretrained(
            LENS_REPO, filename=LENS_FILENAME, revision=LENS_REVISION
        )
        if int(lens.d_model) != 5120 or int(lens.d_model) != int(model.d_model):
            raise RuntimeError(
                f"pinned residual width mismatch: lens={lens.d_model}, model={model.d_model}"
            )
        if [int(layer) for layer in lens.source_layers] != list(range(63)):
            raise RuntimeError("pinned lens no longer exposes exactly source layers 0-62")
    resolved = model_info(MODEL_ID, revision=MODEL_REVISION).sha
    metadata = {
        "model_id": MODEL_ID,
        "model_revision_requested": MODEL_REVISION,
        "model_revision_resolved": resolved,
        "tokenizer_revision": resolved,
        "lens_repo": LENS_REPO if with_lens else None,
        "lens_revision": LENS_REVISION if with_lens else None,
        "lens_filename": LENS_FILENAME if with_lens else None,
        "lens_code_commit": JLENS_COMMIT if with_lens else None,
        "lens_source_layers": [int(layer) for layer in lens.source_layers] if lens else [],
        "dtype": "bfloat16",
        "gpu_actual": torch.cuda.get_device_name(0),
        "torch_version": str(torch.__version__),
        "transformers_version": transformers.__version__,
    }
    return model, lens, metadata


def _tokenize_payload(source: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    import transformers

    from jspace_policy.strategic_trajectories import (
        continuation_token_ids,
        verify_dataset_payload,
    )

    _validate_config_pins(config)
    if not _hash_valid(source):
        raise RuntimeError("source dataset hash mismatch")
    verify_dataset_payload(source, config)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    payload = json.loads(json.dumps(source))
    for row in payload["rows"]:
        task_text = str(row["prompt"])
        system_key = f"{row['reasoning_mode']}_system_prompt"
        rendered = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": config["generation"][system_key]},
                {"role": "user", "content": task_text},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        action_ids = continuation_token_ids(
            lambda text: tokenizer.encode(text, add_special_tokens=False),
            rendered + str(config["generation"]["final_prefix"]),
        )
        row["task_text"] = task_text
        row["task_sha256"] = row["prompt_sha256"]
        row["prompt"] = rendered
        row["prompt_sha256"] = hashlib.sha256(rendered.encode()).hexdigest()
        row["prompt_token_ids"] = tokenizer.encode(rendered, add_special_tokens=False)
        row["label_token_ids"] = action_ids
        row["sequence_length"] = len(row["prompt_token_ids"])
    payload["status"] = "tokenized_and_frozen_before_behavior"
    payload["source_content_sha256"] = source["content_sha256"]
    payload["tokenizer_revision"] = MODEL_REVISION
    payload["created_at"] = datetime.now(UTC).isoformat()
    payload.pop("content_sha256", None)
    payload["content_sha256"] = _canonical_sha256(payload)
    cache.commit()
    return payload


@app.function(image=image, volumes={"/cache": cache}, cpu=2.0, memory=4096, timeout=900)
def tokenize_remote(source: dict[str, Any], config: dict[str, Any]) -> str:
    return json.dumps(_tokenize_payload(source, config), sort_keys=True)


def _trim_special_tokens(token_ids: list[int], tokenizer: Any) -> list[int]:
    special = {tokenizer.pad_token_id, tokenizer.eos_token_id}
    while token_ids and token_ids[-1] in special:
        token_ids.pop()
    return token_ids


def _generate_behavior(
    model: Any, dataset: dict[str, Any], config: dict[str, Any]
) -> list[dict[str, Any]]:
    import torch

    from jspace_policy.strategic_trajectories import parse_output

    output = []
    for row in sorted(dataset["rows"], key=lambda item: item["condition_id"]):
        input_ids = torch.tensor([row["prompt_token_ids"]], dtype=torch.long, device="cuda")
        maximum = int(config["generation"][f"{row['reasoning_mode']}_max_new_tokens"])
        with torch.inference_mode():
            sequence = model._hf_model.generate(
                input_ids=input_ids,
                do_sample=False,
                max_new_tokens=maximum,
                pad_token_id=model.tokenizer.pad_token_id,
                eos_token_id=model.tokenizer.eos_token_id,
                use_cache=True,
            )
        generated_ids = _trim_special_tokens(
            [int(token) for token in sequence[0, input_ids.shape[1] :].cpu()],
            model.tokenizer,
        )
        text = model.tokenizer.decode(generated_ids, skip_special_tokens=True)
        parsed = parse_output(text, row["reasoning_mode"])
        output.append(
            {
                "condition_id": row["condition_id"],
                "instance_id": row["instance_id"],
                "pair_id": row["pair_id"],
                "pair_kind": row["pair_kind"],
                "pair_member": row["pair_member"],
                "framing": row["framing"],
                "reasoning_mode": row["reasoning_mode"],
                "expected_label": row["expected_label"],
                "generated_token_ids": generated_ids,
                "generated_text": text,
                "generated_tokens": len(generated_ids),
                "parseable": parsed["parseable"],
                "final_label": parsed["label"],
                "correct": parsed["label"] == row["expected_label"],
                "reasoning_sentence_count": parsed["reasoning_sentence_count"],
                "first_reasoning_sentence": parsed["first_reasoning_sentence"],
            }
        )
        del input_ids, sequence
    return output


@app.function(
    image=image,
    volumes={"/cache": cache},
    cpu=4.0,
    memory=32768,
    gpu="A100-80GB",
    timeout=1200,
    max_containers=1,
    retries=0,
)
def behavior_remote(dataset: dict[str, Any], config: dict[str, Any], git_commit: str) -> str:
    import torch

    from jspace_policy.strategic_trajectories import behavior_summary

    _validate_config_pins(config)
    if not _hash_valid(dataset):
        raise RuntimeError("tokenized dataset hash mismatch")
    started = time.perf_counter()
    model, _, metadata = _load_model(with_lens=False)
    rows = _generate_behavior(model, dataset, config)
    summary = behavior_summary(rows)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    cache.commit()
    return json.dumps(
        {
            "metadata": {
                "schema_version": 1,
                "run_id": uuid.uuid4().hex,
                "created_at": datetime.now(UTC).isoformat(),
                "git_commit": git_commit,
                "dataset_sha256": dataset["content_sha256"],
                "source_dataset_sha256": dataset["source_content_sha256"],
                "config_sha256": _canonical_sha256(config),
                "elapsed_seconds": elapsed,
                **metadata,
            },
            "summary": summary,
            "rows": rows,
        },
        allow_nan=False,
        sort_keys=True,
    )


def _top_readout_batch(
    model: Any, lens: Any, layer: int, residuals: Any, k: int
) -> list[list[dict[str, Any]]]:
    import torch

    if residuals.ndim == 1:
        residuals = residuals.unsqueeze(0)
    jacobian = lens.jacobians[layer].to(residuals.device)
    with torch.inference_mode():
        transported = residuals.float() @ jacobian.float().T
        logits = model.unembed(transported).float()
        top = logits.topk(k, dim=-1)
        normalization = torch.logsumexp(logits, dim=-1, keepdim=True)
        top_log_probs = top.values - normalization
    top_ids = top.indices.detach().cpu()
    top_scores = top.values.detach().cpu()
    top_logs = top_log_probs.detach().cpu()
    rows = []
    for batch_index in range(residuals.shape[0]):
        rows.append(
            [
                {
                    "rank": rank + 1,
                    "token_id": int(token_id),
                    "token": model.tokenizer.decode([int(token_id)]),
                    "raw_score": float(score),
                    "log_probability": float(log_probability),
                }
                for rank, (token_id, score, log_probability) in enumerate(
                    zip(
                        top_ids[batch_index],
                        top_scores[batch_index],
                        top_logs[batch_index],
                        strict=True,
                    )
                )
            ]
        )
    del jacobian, transported, logits, top, top_log_probs
    return rows


def _pre_final_trace_index(
    generated_ids: list[int], tokenizer: Any, label_token_ids: dict[str, int]
) -> int | None:
    legal = set(map(int, label_token_ids.values()))
    for split in range(1, len(generated_ids)):
        prefix = tokenizer.decode(generated_ids[:split], skip_special_tokens=True)
        if prefix.endswith("FINAL:") and generated_ids[split] in legal:
            # Trace index 0 is the final prompt; generated token j has trace index j+1.
            return split
    return None


def _mechanistic_row(
    model: Any,
    lens: Any,
    dataset_row: dict[str, Any],
    behavior_row: dict[str, Any],
    config: dict[str, Any],
    residual_root: Path,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from jlens import ActivationRecorder

    prompt_ids = list(map(int, dataset_row["prompt_token_ids"]))
    generated_ids = list(map(int, behavior_row["generated_token_ids"]))
    sequence_ids = prompt_ids + generated_ids
    input_ids = torch.tensor([sequence_ids], dtype=torch.long, device="cuda")
    all_layers = list(range(63))
    selected_layers = list(map(int, config["trajectory"]["layers"]))
    with torch.inference_mode(), ActivationRecorder(model.layers, at=all_layers) as recorder:
        outputs = model._hf_model(input_ids)

    final_prompt_position = len(prompt_ids) - 1
    trace_positions = [final_prompt_position, *range(len(prompt_ids), len(sequence_ids))]
    residual_array = np.stack(
        [
            recorder.activations[layer][0, trace_positions]
            .detach()
            .cpu()
            .to(torch.float16)
            .numpy()
            for layer in selected_layers
        ],
        axis=1,
    )
    residual_path = residual_root / f"{dataset_row['condition_id']}.npy"
    if residual_path.exists():
        raise RuntimeError(f"refusing to overwrite residual artifact: {residual_path}")
    residual_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(residual_path, residual_array, allow_pickle=False)

    candidate_ids = [int(dataset_row["label_token_ids"][label]) for label in "ABC"]
    readouts_by_layer = {}
    top_k = int(config["trajectory"]["top_k"])
    for layer in selected_layers:
        residuals = recorder.activations[layer][0, trace_positions].detach()
        readouts_by_layer[layer] = _top_readout_batch(
            model, lens, layer, residuals, top_k
        )
    traces = []
    for trace_index, sequence_position in enumerate(trace_positions):
        model_logits = outputs.logits[0, sequence_position].float()
        legal_logits = model_logits[candidate_ids]
        legal_log_probs = legal_logits.log_softmax(-1)
        layer_readouts = {
            str(layer): readouts_by_layer[layer][trace_index] for layer in selected_layers
        }
        generated_index = trace_index - 1
        traces.append(
            {
                "trace_index": trace_index,
                "sequence_position": sequence_position,
                "kind": "final_prompt" if trace_index == 0 else "generated_token",
                "generated_index": generated_index if generated_index >= 0 else None,
                "surface_token_id": (
                    generated_ids[generated_index] if generated_index >= 0 else None
                ),
                "surface_token": (
                    model.tokenizer.decode([generated_ids[generated_index]])
                    if generated_index >= 0
                    else None
                ),
                "legal_action_logits": dict(
                    zip("ABC", map(float, legal_logits.cpu()), strict=True)
                ),
                "legal_action_log_probs": dict(
                    zip("ABC", map(float, legal_log_probs.cpu()), strict=True)
                ),
                "layers": layer_readouts,
            }
        )

    final_prompt_readout = []
    final_k = int(config["trajectory"]["final_prompt_top_k"])
    for layer in all_layers:
        residual = recorder.activations[layer][0, final_prompt_position].detach()
        final_prompt_readout.append(
            {
                "layer": layer,
                "top_tokens": _top_readout_batch(
                    model, lens, layer, residual, final_k
                )[0],
            }
        )
    residual_sha256 = hashlib.sha256(residual_path.read_bytes()).hexdigest()
    pre_final = _pre_final_trace_index(
        generated_ids, model.tokenizer, dataset_row["label_token_ids"]
    )
    del input_ids, outputs, recorder
    return {
        "condition_id": dataset_row["condition_id"],
        "instance_id": dataset_row["instance_id"],
        "pair_id": dataset_row["pair_id"],
        "pair_kind": dataset_row["pair_kind"],
        "pair_member": dataset_row["pair_member"],
        "framing": dataset_row["framing"],
        "reasoning_mode": dataset_row["reasoning_mode"],
        "expected_label": dataset_row["expected_label"],
        "decisive_response": dataset_row["decisive_response"],
        "generated_text": behavior_row["generated_text"],
        "synchronization": {
            "final_prompt_trace_index": 0,
            "pre_final_trace_index": pre_final,
            "first_completed_reasoning_sentence": behavior_row["first_reasoning_sentence"],
        },
        "residual_artifact": {
            "volume": "jspace-v2-e2-trajectory-artifacts",
            "path": str(residual_path),
            "sha256": residual_sha256,
            "shape": list(residual_array.shape),
            "dtype": str(residual_array.dtype),
            "layer_axis": selected_layers,
        },
        "trace": traces,
        "all_layer_final_prompt": final_prompt_readout,
    }


def _human_transcript(row: dict[str, Any]) -> str:
    lines = [
        f"CONDITION {row['condition_id']}",
        f"PAIR {row['pair_id']} member={row['pair_member']} kind={row['pair_kind']}",
        f"FRAMING {row['framing']} | MODE {row['reasoning_mode']}",
        f"EXPECTED {row['expected_label']} | DECISIVE {row['decisive_response']}",
        "",
        "SURFACE",
        row["generated_text"],
        "",
        "TRAJECTORY",
    ]
    for trace in row["trace"]:
        token = (
            "<FINAL_PROMPT>"
            if trace["surface_token"] is None
            else repr(trace["surface_token"])
        )
        lines.append(f"t={trace['trace_index']} token={token}")
        for layer, tokens in trace["layers"].items():
            words = " | ".join(item["token"].replace("\n", "\\n") for item in tokens)
            lines.append(f"L{layer}: {words}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


@app.function(
    image=image,
    volumes={"/cache": cache, "/artifacts": artifacts},
    cpu=8.0,
    memory=65536,
    gpu="A100-80GB",
    timeout=3000,
    max_containers=1,
    retries=0,
)
def mechanistic_remote(
    dataset: dict[str, Any],
    behavior_result: dict[str, Any],
    config: dict[str, Any],
    git_commit: str,
) -> str:
    import torch

    _validate_config_pins(config)
    if not _hash_valid(dataset):
        raise RuntimeError("tokenized dataset hash mismatch")
    if not behavior_result["summary"]["gate_pass"]:
        raise RuntimeError("behavioral gate failed; mechanistic activations remain unopened")
    if behavior_result["metadata"]["dataset_sha256"] != dataset["content_sha256"]:
        raise RuntimeError("behavior result does not match the frozen tokenized dataset")
    started = time.perf_counter()
    run_id = uuid.uuid4().hex
    model, lens, metadata = _load_model(with_lens=True)
    behavior_by_id = {row["condition_id"]: row for row in behavior_result["rows"]}
    remote_root = Path(f"/artifacts/runs/{run_id}")
    residual_root = remote_root / "residuals"
    rows = []
    for dataset_row in sorted(dataset["rows"], key=lambda item: item["condition_id"]):
        rows.append(
            _mechanistic_row(
                model,
                lens,
                dataset_row,
                behavior_by_id[dataset_row["condition_id"]],
                config,
                residual_root,
            )
        )
    payload = {
        "metadata": {
            "schema_version": 1,
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "git_commit": git_commit,
            "dataset_sha256": dataset["content_sha256"],
            "behavior_run_id": behavior_result["metadata"]["run_id"],
            "config_sha256": _canonical_sha256(config),
            "interpretation": "observational human-readable J-lens trajectory only",
            **metadata,
        },
        "rows": rows,
    }
    transcript_root = remote_root / "transcripts"
    transcript_root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        (transcript_root / f"{row['condition_id']}.txt").write_text(
            _human_transcript(row), encoding="utf-8"
        )
    payload_path = remote_root / "mechanistic.json.gz"
    with gzip.open(payload_path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False, sort_keys=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    payload_sha256 = hashlib.sha256(payload_path.read_bytes()).hexdigest()
    manifest = {
        "metadata": {
            **payload["metadata"],
            "elapsed_seconds": elapsed,
        },
        "artifact": {
            "volume": "jspace-v2-e2-trajectory-artifacts",
            "root": str(remote_root),
            "payload_path": str(payload_path),
            "payload_sha256": payload_sha256,
            "payload_bytes": payload_path.stat().st_size,
            "residual_files": len(rows),
            "transcript_files": len(rows),
        },
    }
    artifacts.commit()
    cache.commit()
    return json.dumps(manifest, allow_nan=False, sort_keys=True)


def _admit(config: dict[str, Any], stage: str, seconds: int, memory_gib: int) -> Path:
    ledger = RESULT_ROOT / "cost_ledger.jsonl"
    estimate = estimate_cost(
        str(config["execution"]["gpu"]),
        seconds,
        cpu_cores=8,
        memory_gib=memory_gib,
    )
    admit_run(
        ledger,
        estimate,
        study_limit_usd=float(config["execution"]["hard_cost_limit_usd"]),
    )
    return ledger


def _record_cost(
    ledger: Path, payload: dict[str, Any], *, stage: str, memory_gib: int
) -> None:
    measured = estimate_cost(
        "A100-80GB",
        float(payload["metadata"]["elapsed_seconds"]),
        cpu_cores=8,
        memory_gib=memory_gib,
    )
    append_ledger(ledger, measured, run_id=payload["metadata"]["run_id"], stage=stage)


@app.local_entrypoint()
def freeze_dataset() -> None:
    config = _load_json(CONFIG_PATH)
    _validate_config_pins(config)
    source = _load_json(SOURCE_DATASET_PATH)
    payload = json.loads(tokenize_remote.remote(source, config))
    _write_new(TOKENIZED_DATASET_PATH, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(TOKENIZED_DATASET_PATH),
                "rows": len(payload["rows"]),
                "content_sha256": payload["content_sha256"],
            },
            indent=2,
        )
    )


@app.local_entrypoint()
def behavior() -> None:
    config = _load_json(CONFIG_PATH)
    _validate_config_pins(config)
    dataset = _load_json(TOKENIZED_DATASET_PATH)
    output = RESULT_ROOT / "raw" / "behavior.json"
    ledger = _admit(config, "behavior", 1200, 32)
    parsed = json.loads(behavior_remote.remote(dataset, config, _git_head()))
    _write_new(output, json.dumps(parsed, indent=2, sort_keys=True) + "\n")
    _record_cost(ledger, parsed, stage="e2_trajectory_behavior", memory_gib=32)
    print(json.dumps(parsed["summary"], indent=2, sort_keys=True))


@app.local_entrypoint()
def mechanistic() -> None:
    config = _load_json(CONFIG_PATH)
    _validate_config_pins(config)
    dataset = _load_json(TOKENIZED_DATASET_PATH)
    behavior_result = _load_json(RESULT_ROOT / "raw" / "behavior.json")
    output = RESULT_ROOT / "raw" / "mechanistic_manifest.json"
    ledger = _admit(config, "mechanistic", 3000, 64)
    parsed = json.loads(
        mechanistic_remote.remote(dataset, behavior_result, config, _git_head())
    )
    _write_new(output, json.dumps(parsed, indent=2, sort_keys=True) + "\n")
    _record_cost(ledger, parsed, stage="e2_trajectory_mechanistic", memory_gib=64)
    print(json.dumps(parsed["artifact"], indent=2, sort_keys=True))
