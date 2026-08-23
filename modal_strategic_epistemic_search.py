"""Budget-gated V4 strategic epistemic search.

Commands:
    modal run modal_strategic_epistemic_search.py::freeze_dataset
    modal run modal_strategic_epistemic_search.py::behavior
    modal run modal_strategic_epistemic_search.py::mechanistic

The first mechanistic pass is observational.  Causal entrypoints are intentionally
added only after discovery layers and matched pairs have been frozen.
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

CONFIG_PATH = Path("configs/v4/strategic_epistemic_search/experiment.json")
DATASET_PATH = Path("configs/v4/strategic_epistemic_search/dataset.json")
RESULT_ROOT = Path("results/v4_strategic_epistemic_search")

MODEL_ID = "Qwen/Qwen3.6-27B"
MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
JLENS_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
LENS_FILENAME = (
    "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt"
)

app = modal.App("jspace-v4-strategic-epistemic-search")
cache = modal.Volume.from_name("jspace-hf-cache", create_if_missing=True)
artifacts = modal.Volume.from_name(
    "jspace-v4-strategic-epistemic-artifacts", create_if_missing=True
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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_new(path: Path, value: str) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite prospective artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _worktree_sha256() -> str:
    diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"])
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"], text=True
    ).splitlines()
    digest = hashlib.sha256(diff)
    for name in sorted(untracked):
        path = Path(name)
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _validate_config(config: dict[str, Any]) -> None:
    if config["model"]["id"] != MODEL_ID or config["model"]["revision"] != MODEL_REVISION:
        raise RuntimeError("model pins changed")
    if float(config["execution"]["hard_cost_limit_usd"]) > 18.5:
        raise RuntimeError("V4 cost ceiling may not exceed USD 18.50")
    layers = list(map(int, config["mechanistic"]["layers"]))
    if not layers or min(layers) < 0 or max(layers) > 62:
        raise RuntimeError("mechanistic layer list is invalid")


def _hash_valid(payload: dict[str, Any]) -> bool:
    claimed = payload.get("content_sha256")
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    return claimed == _canonical_sha256(body)


def _continuation_id(tokenizer: Any, prompt: str, answer: str) -> int:
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    full = tokenizer.encode(prompt + answer, add_special_tokens=False)
    if full[: len(prefix)] == prefix and len(full) == len(prefix) + 1:
        return int(full[-1])
    raise ValueError(f"{answer!r} is not one token after the frozen Answer: prefix")


def _find_last_subsequence(sequence: list[int], needle: list[int]) -> int:
    matches = [
        index + len(needle) - 1
        for index in range(len(sequence) - len(needle) + 1)
        if sequence[index : index + len(needle)] == needle
    ]
    if not matches:
        raise ValueError(f"marker token sequence not found: {needle}")
    return matches[-1]


def _tokenize_dataset(dataset: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    import transformers

    from jspace_policy.strategic_epistemic_search import verify_dataset_payload

    verify_dataset_payload(dataset, config)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    payload = json.loads(json.dumps(dataset))
    for row in payload["rows"]:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        token_ids = list(map(int, tokenizer.encode(rendered, add_special_tokens=False)))
        row["task_text"] = row["prompt"]
        row["prompt"] = rendered
        row["prompt_token_ids"] = token_ids
        row["candidate_token_ids"] = [
            _continuation_id(tokenizer, rendered, label) for label in "ABC"
        ]
        row["expected_action_token_id"] = row["candidate_token_ids"][
            "ABC".index(row["expected_action"])
        ]
        row["marker_positions"] = {
            name: _find_last_subsequence(
                token_ids,
                list(
                    map(
                        int,
                        tokenizer.encode(text, add_special_tokens=False),
                    )
                ),
            )
            for name, text in row["marker_text"].items()
        }
        row["marker_positions"]["final_prompt"] = len(token_ids) - 1
        row["sequence_length"] = len(token_ids)
    payload["status"] = "tokenized_at_remote_execution"
    payload["source_content_sha256"] = dataset["content_sha256"]
    payload["content_sha256"] = _canonical_sha256(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )
    cache.commit()
    return payload


def _load_model(*, with_lens: bool) -> tuple[Any, Any | None, dict[str, Any]]:
    import torch
    import transformers
    from huggingface_hub import model_info
    from jlens import JacobianLens, from_hf

    torch.manual_seed(82326)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).cuda()
    hf_model.eval()
    model = from_hf(hf_model, tokenizer)
    lens = None
    if with_lens:
        lens = JacobianLens.from_pretrained(
            LENS_REPO, filename=LENS_FILENAME, revision=LENS_REVISION
        )
        if int(lens.d_model) != int(model.d_model):
            raise RuntimeError("model/lens width mismatch")
    resolved = model_info(MODEL_ID, revision=MODEL_REVISION).sha
    metadata = {
        "model_id": MODEL_ID,
        "model_revision_requested": MODEL_REVISION,
        "model_revision_resolved": resolved,
        "lens_repo": LENS_REPO if with_lens else None,
        "lens_revision": LENS_REVISION if with_lens else None,
        "lens_filename": LENS_FILENAME if with_lens else None,
        "lens_code_commit": JLENS_COMMIT if with_lens else None,
        "dtype": "bfloat16",
        "gpu_actual": torch.cuda.get_device_name(0),
        "torch_version": str(torch.__version__),
        "transformers_version": transformers.__version__,
    }
    return model, lens, metadata


def _left_padded_batch(rows: list[dict[str, Any]]) -> tuple[Any, Any, int]:
    import torch

    width = max(len(row["prompt_token_ids"]) for row in rows)
    input_ids = torch.zeros((len(rows), width), dtype=torch.long, device="cuda")
    attention = torch.zeros_like(input_ids)
    for index, row in enumerate(rows):
        tokens = torch.tensor(row["prompt_token_ids"], dtype=torch.long, device="cuda")
        input_ids[index, width - len(tokens) :] = tokens
        attention[index, width - len(tokens) :] = 1
    return input_ids, attention, width


def _behavior_rows(
    model: Any, rows: list[dict[str, Any]], batch_size: int
) -> list[dict[str, Any]]:
    import torch

    output: list[dict[str, Any]] = []
    ordered = sorted(rows, key=lambda row: (row["sequence_length"], row["condition_id"]))
    for start in range(0, len(ordered), batch_size):
        part = ordered[start : start + batch_size]
        input_ids, attention, _ = _left_padded_batch(part)
        with torch.inference_mode():
            logits = model._hf_model(
                input_ids, attention_mask=attention, logits_to_keep=1
            ).logits[:, -1].float()
        for index, row in enumerate(part):
            final = logits[index]
            candidates = list(map(int, row["candidate_token_ids"]))
            candidate_logits = final[candidates]
            legal_choice = int(candidate_logits.argmax().cpu())
            top1 = int(final.argmax().cpu())
            output.append(
                {
                    "condition_id": row["condition_id"],
                    "pair_id": row["pair_id"],
                    "pair_type": row["pair_type"],
                    "side": row["side"],
                    "split": row["split"],
                    "frame": row["frame"],
                    "expected_action": row["expected_action"],
                    "top1_token_id": top1,
                    "top1_text": model.tokenizer.decode([top1]),
                    "formatting_compliant": top1 in candidates,
                    "correct": top1 == int(row["expected_action_token_id"]),
                    "legal_choice": "ABC"[legal_choice],
                    "legal_choice_correct": "ABC"[legal_choice] == row["expected_action"],
                    "legal_action_logits": {
                        label: float(value)
                        for label, value in zip(
                            "ABC", candidate_logits.detach().cpu(), strict=True
                        )
                    },
                }
            )
        del input_ids, attention, logits
    return sorted(output, key=lambda row: row["condition_id"])


def _behavior_summary(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    overall = sum(row["correct"] for row in rows) / len(rows)
    cells = {
        f"{split}:{frame}:{pair_type}": sum(
            row["correct"]
            for row in rows
            if row["split"] == split
            and row["frame"] == frame
            and row["pair_type"] == pair_type
        )
        / sum(
            row["split"] == split
            and row["frame"] == frame
            and row["pair_type"] == pair_type
            for row in rows
        )
        for split in sorted({row["split"] for row in rows})
        for frame in sorted({row["frame"] for row in rows})
        for pair_type in sorted({row["pair_type"] for row in rows})
    }
    minimum_cell = min(cells.values())
    gate = (
        overall >= float(config["behavior"]["minimum_overall_accuracy"])
        and minimum_cell >= float(config["behavior"]["minimum_cell_accuracy"])
    )
    return {
        "n_rows": len(rows),
        "overall_accuracy": overall,
        "minimum_cell_accuracy": minimum_cell,
        "cell_accuracy": cells,
        "gate_pass": gate,
    }


@app.function(
    image=image,
    volumes={"/cache": cache},
    cpu=4.0,
    memory=32768,
    gpu="A100-80GB",
    timeout=1800,
    max_containers=1,
    retries=0,
)
def behavior_remote(
    dataset: dict[str, Any], config: dict[str, Any], code_metadata: dict[str, str]
) -> str:
    import torch

    _validate_config(config)
    if not _hash_valid(dataset):
        raise RuntimeError("source dataset hash mismatch")
    started = time.perf_counter()
    tokenized = _tokenize_dataset(dataset, config)
    model, _, metadata = _load_model(with_lens=False)
    rows = _behavior_rows(model, tokenized["rows"], int(config["behavior"]["batch_size"]))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    cache.commit()
    return json.dumps(
        {
            "metadata": {
                "schema_version": 1,
                "run_id": uuid.uuid4().hex,
                "created_at": datetime.now(UTC).isoformat(),
                "source_dataset_sha256": dataset["content_sha256"],
                "tokenized_dataset_sha256": tokenized["content_sha256"],
                "config_sha256": _canonical_sha256(config),
                "elapsed_seconds": elapsed,
                **code_metadata,
                **metadata,
            },
            "summary": _behavior_summary(rows, config),
            "rows": rows,
        },
        allow_nan=False,
        sort_keys=True,
    )


def _top_readout(model: Any, lens: Any, layer: int, residuals: Any, k: int) -> list[Any]:
    import torch

    jacobian = lens.jacobians[layer].to(residuals.device)
    with torch.inference_mode():
        transported = residuals.float() @ jacobian.float().T
        logits = model.unembed(transported).float()
        top = logits.topk(k, dim=-1)
    ids = top.indices.detach().cpu()
    scores = top.values.detach().cpu()
    result = []
    for batch_index in range(residuals.shape[0]):
        result.append(
            [
                {
                    "rank": rank + 1,
                    "token_id": int(token_id),
                    "token": model.tokenizer.decode([int(token_id)]),
                    "score": float(score),
                }
                for rank, (token_id, score) in enumerate(
                    zip(ids[batch_index], scores[batch_index], strict=True)
                )
            ]
        )
    del jacobian, transported, logits, top
    return result


@app.function(
    image=image,
    volumes={"/cache": cache, "/artifacts": artifacts},
    cpu=8.0,
    memory=65536,
    gpu="A100-80GB",
    timeout=3600,
    max_containers=1,
    retries=0,
)
def mechanistic_remote(
    dataset: dict[str, Any],
    behavior_result: dict[str, Any],
    config: dict[str, Any],
    code_metadata: dict[str, str],
) -> str:
    import numpy as np
    import torch
    from jlens import ActivationRecorder

    _validate_config(config)
    if not behavior_result["summary"]["gate_pass"]:
        raise RuntimeError("behavior gate failed; mechanistic run refused")
    if behavior_result["metadata"]["source_dataset_sha256"] != dataset["content_sha256"]:
        raise RuntimeError("behavior result does not match source dataset")
    started = time.perf_counter()
    tokenized = _tokenize_dataset(dataset, config)
    if behavior_result["metadata"]["tokenized_dataset_sha256"] != tokenized["content_sha256"]:
        raise RuntimeError("tokenized dataset changed after behavior")
    model, lens, metadata = _load_model(with_lens=True)
    layers = list(map(int, config["mechanistic"]["layers"]))
    markers = list(config["mechanistic"]["marker_names"])
    top_k = int(config["mechanistic"]["j_lens_top_k"])
    run_id = uuid.uuid4().hex
    remote_root = Path(f"/artifacts/runs/{run_id}")
    residual_root = remote_root / "residuals"
    residual_root.mkdir(parents=True, exist_ok=True)
    output_rows = []
    ordered = sorted(tokenized["rows"], key=lambda row: row["condition_id"])
    batch_size = int(config["behavior"]["batch_size"])
    for start in range(0, len(ordered), batch_size):
        part = ordered[start : start + batch_size]
        input_ids, attention, width = _left_padded_batch(part)
        with torch.inference_mode(), ActivationRecorder(model.layers, at=layers) as recorder:
            outputs = model._hf_model(input_ids, attention_mask=attention)
        for batch_index, row in enumerate(part):
            offset = width - len(row["prompt_token_ids"])
            positions = [offset + int(row["marker_positions"][name]) for name in markers]
            residual_array = np.stack(
                [
                    recorder.activations[layer][batch_index, positions]
                    .detach()
                    .cpu()
                    .to(torch.float16)
                    .numpy()
                    for layer in layers
                ],
                axis=1,
            )
            residual_path = residual_root / f"{row['condition_id']}.npy"
            np.save(residual_path, residual_array, allow_pickle=False)
            readouts = {}
            for layer in layers:
                residuals = recorder.activations[layer][batch_index, positions].detach()
                layer_rows = _top_readout(model, lens, layer, residuals, top_k)
                readouts[str(layer)] = {
                    marker: layer_rows[index] for index, marker in enumerate(markers)
                }
            final_logits = outputs.logits[batch_index, -1].float()
            candidate_ids = list(map(int, row["candidate_token_ids"]))
            output_rows.append(
                {
                    "condition_id": row["condition_id"],
                    "pair_id": row["pair_id"],
                    "pair_type": row["pair_type"],
                    "side": row["side"],
                    "split": row["split"],
                    "frame": row["frame"],
                    "expected_action": row["expected_action"],
                    "winner": row["winner"],
                    "runner_up": row["runner_up"],
                    "decisive_response": row["decisive_response"],
                    "margin": row["margin"],
                    "values": row["values"],
                    "policy": row["policy"],
                    "payoffs": row["payoffs"],
                    "costs": row["costs"],
                    "legal_action_logits": {
                        label: float(value)
                        for label, value in zip(
                            "ABC", final_logits[candidate_ids].detach().cpu(), strict=True
                        )
                    },
                    "residual_artifact": {
                        "path": str(residual_path),
                        "sha256": hashlib.sha256(residual_path.read_bytes()).hexdigest(),
                        "shape": list(residual_array.shape),
                        "dtype": str(residual_array.dtype),
                        "marker_axis": markers,
                        "layer_axis": layers,
                    },
                    "j_lens": readouts,
                }
            )
        del input_ids, attention, outputs, recorder
    payload = {
        "metadata": {
            "schema_version": 1,
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "source_dataset_sha256": dataset["content_sha256"],
            "tokenized_dataset_sha256": tokenized["content_sha256"],
            "behavior_run_id": behavior_result["metadata"]["run_id"],
            "config_sha256": _canonical_sha256(config),
            **code_metadata,
            **metadata,
        },
        "rows": sorted(output_rows, key=lambda row: row["condition_id"]),
    }
    payload_path = remote_root / "mechanistic.json.gz"
    with gzip.open(payload_path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False, sort_keys=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    artifacts.commit()
    cache.commit()
    return json.dumps(
        {
            "metadata": {**payload["metadata"], "elapsed_seconds": elapsed},
            "artifact": {
                "volume": "jspace-v4-strategic-epistemic-artifacts",
                "root": str(remote_root),
                "payload_path": str(payload_path),
                "payload_sha256": hashlib.sha256(payload_path.read_bytes()).hexdigest(),
                "payload_bytes": payload_path.stat().st_size,
                "residual_files": len(output_rows),
            },
        },
        allow_nan=False,
        sort_keys=True,
    )


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
    ledger: Path, config: dict[str, Any], payload: dict[str, Any], stage: str, memory_gib: int
) -> None:
    measured = estimate_cost(
        str(config["execution"]["gpu"]),
        float(payload["metadata"]["elapsed_seconds"]),
        cpu_cores=8,
        memory_gib=memory_gib,
    )
    append_ledger(
        ledger,
        measured,
        run_id=str(payload["metadata"]["run_id"]),
        stage=stage,
    )


@app.local_entrypoint()
def freeze_dataset() -> None:
    from jspace_policy.strategic_epistemic_search import (
        dataset_payload,
        verify_dataset_payload,
    )

    config = _load_json(CONFIG_PATH)
    _validate_config(config)
    payload = dataset_payload(config)
    verify_dataset_payload(payload, config)
    _write_new(DATASET_PATH, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {DATASET_PATH} with {len(payload['rows'])} rows")


@app.local_entrypoint()
def behavior() -> None:
    config = _load_json(CONFIG_PATH)
    dataset = _load_json(DATASET_PATH)
    _validate_config(config)
    ledger = _admit(config, "behavior", 1800, 32)
    code_metadata = {"git_commit": _git_head(), "worktree_sha256": _worktree_sha256()}
    payload = json.loads(behavior_remote.remote(dataset, config, code_metadata))
    target = RESULT_ROOT / "raw/behavior.json"
    _write_new(target, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _record_cost(ledger, config, payload, "behavior", 32)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


@app.local_entrypoint()
def mechanistic() -> None:
    config = _load_json(CONFIG_PATH)
    dataset = _load_json(DATASET_PATH)
    behavior_result = _load_json(RESULT_ROOT / "raw/behavior.json")
    _validate_config(config)
    ledger = _admit(config, "mechanistic", 3600, 64)
    code_metadata = {"git_commit": _git_head(), "worktree_sha256": _worktree_sha256()}
    payload = json.loads(
        mechanistic_remote.remote(dataset, behavior_result, config, code_metadata)
    )
    target = RESULT_ROOT / "raw/mechanistic_manifest.json"
    _write_new(target, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _record_cost(ledger, config, payload, "mechanistic", 64)
    print(json.dumps(payload, indent=2, sort_keys=True))
