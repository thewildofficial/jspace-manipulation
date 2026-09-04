"""V5-RBG-6 GPU-only full-action-trajectory execution."""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

from jspace_policy.budget import admit_run, append_ledger, estimate_cost
from jspace_policy.full_action_trajectory import canonical_sha256, verify_dataset_payload

CONFIG_PATH = Path("configs/v5/full_action_trajectory/experiment.json")
DATASET_PATH = Path("configs/v5/full_action_trajectory/dataset.json")
PREPARED_PATH = Path("artifacts/processed/v5_full_action_trajectory_prepared.json")
RESULT_ROOT = Path("results/v5_full_action_trajectory")
REMOTE_CONFIG = Path("/root/experiment.json")
REMOTE_DATASET = Path("/root/dataset.json")
REMOTE_PREPARED = Path("/root/prepared.json")
MODEL_ID = "Qwen/Qwen3.6-27B"
MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"

app = modal.App("jspace-v5-full-action-trajectory")
cache = modal.Volume.from_name("jspace-hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .uv_pip_install(
        "torch>=2.8",
        "transformers>=5.5",
        "huggingface_hub>=0.34",
    )
    .env({"HF_HOME": "/cache/huggingface", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir("src/jspace_policy", remote_path="/root/jspace_policy")
    .add_local_file(str(CONFIG_PATH), remote_path=str(REMOTE_CONFIG))
    .add_local_file(str(DATASET_PATH), remote_path=str(REMOTE_DATASET))
    .add_local_file(str(PREPARED_PATH), remote_path=str(REMOTE_PREPARED))
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_new(path: Path, value: str) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite prospective artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _validate(
    config: dict[str, Any], dataset: dict[str, Any], prepared: dict[str, Any]
) -> None:
    if config["status"] != (
        "analysis_plan_frozen_after_rbg4_before_full_trajectory_execution"
    ):
        raise RuntimeError("RBG-6 analysis plan is not frozen")
    if config["model"]["id"] != MODEL_ID:
        raise RuntimeError("model ID changed")
    if config["model"]["revision"] != MODEL_REVISION:
        raise RuntimeError("model revision changed")
    if float(config["execution"]["hard_cost_limit_usd"]) > 2.0:
        raise RuntimeError("RBG-6 cost ceiling may not exceed USD 2")
    verify_dataset_payload(dataset, config)
    prepared_body = {
        key: value
        for key, value in prepared.items()
        if key not in {"content_sha256", "preflight"}
    }
    if prepared["content_sha256"] != canonical_sha256(prepared_body):
        raise RuntimeError("prepared payload hash mismatch")
    if prepared["config_sha256"] != canonical_sha256(config):
        raise RuntimeError("prepared/config hash mismatch")
    if prepared["dataset_sha256"] != dataset["content_sha256"]:
        raise RuntimeError("prepared/dataset hash mismatch")
    if {row["trajectory_id"] for row in prepared["rows"]} != {
        row["trajectory_id"] for row in dataset["rows"]
    }:
        raise RuntimeError("prepared trajectory IDs do not match dataset")


def _left_padded(rows: list[dict[str, Any]], pad_token_id: int) -> tuple[Any, Any]:
    import torch

    width = max(row["sequence_length"] for row in rows)
    input_ids = torch.full(
        (len(rows), width), pad_token_id, dtype=torch.long, device="cuda"
    )
    attention = torch.zeros_like(input_ids)
    for index, row in enumerate(rows):
        tokens = torch.tensor(row["prompt_token_ids"], dtype=torch.long, device="cuda")
        input_ids[index, width - len(tokens) :] = tokens
        attention[index, width - len(tokens) :] = 1
    return input_ids, attention


def _query(
    model: Any,
    rows: list[dict[str, Any]],
    pad_token_id: int,
    batch_size: int,
) -> dict[str, dict[str, Any]]:
    import torch

    output: dict[str, dict[str, Any]] = {}
    ordered = sorted(rows, key=lambda row: (row["sequence_length"], row["query_id"]))
    for start in range(0, len(ordered), batch_size):
        part = ordered[start : start + batch_size]
        input_ids, attention = _left_padded(part, pad_token_id)
        with torch.inference_mode():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention,
                logits_to_keep=1,
            ).logits[:, -1].float()
        for index, row in enumerate(part):
            final = logits[index]
            candidate_ids = list(map(int, row["candidate_token_ids"]))
            candidate_logits = final[candidate_ids]
            choice_index = int(candidate_logits.argmax().cpu())
            top1_id = int(final.argmax().cpu())
            output[row["query_id"]] = {
                "legal_choice": row["candidate_labels"][choice_index],
                "legal_logits": {
                    label: float(value)
                    for label, value in zip(
                        row["candidate_labels"],
                        candidate_logits.detach().cpu(),
                        strict=True,
                    )
                },
                "top1_token_id": top1_id,
                "formatting_compliant": top1_id in candidate_ids,
            }
        del input_ids, attention, logits
    return output


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=8,
    memory=32768,
    volumes={"/cache": cache},
    timeout=1200,
)
def full_action_trajectory_gpu() -> str:
    import torch
    import transformers
    from huggingface_hub import model_info

    config = _load_json(REMOTE_CONFIG)
    dataset = _load_json(REMOTE_DATASET)
    prepared = _load_json(REMOTE_PREPARED)
    _validate(config, dataset, prepared)
    started = time.perf_counter()
    torch.manual_seed(82431)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).cuda()
    model.eval()
    pad_token_id = int(prepared["pad_token_id"])

    first_queries = [row["first_report"] for row in prepared["rows"]]
    first_results = _query(model, first_queries, pad_token_id, 24)
    second_queries = []
    for row in prepared["rows"]:
        first_choice = first_results[row["first_report"]["query_id"]]["legal_choice"]
        second_queries.append(row["second_report_by_first_choice"][first_choice])
    second_results = _query(model, second_queries, pad_token_id, 24)

    dataset_by_id = {row["trajectory_id"]: row for row in dataset["rows"]}
    requested_actions: dict[str, dict[str, Any]] = {}
    routing: dict[str, dict[str, str]] = {}
    for row in prepared["rows"]:
        trajectory_id = row["trajectory_id"]
        first_choice = first_results[row["first_report"]["query_id"]]["legal_choice"]
        second_query = row["second_report_by_first_choice"][first_choice]
        second_choice = second_results[second_query["query_id"]]["legal_choice"]
        dataset_row = dataset_by_id[trajectory_id]
        choices = {
            dataset_row["report_order"][0]: first_choice,
            dataset_row["report_order"][1]: second_choice,
        }
        self_key = f"A:{choices['A']}|B:{choices['B']}"
        keys = {
            "self_generated": self_key,
            "oracle_replay": row["oracle_pair_key"],
            "swapped_replay": row["swapped_pair_key"],
        }
        routing[trajectory_id] = keys
        for key in keys.values():
            query = row["action_by_report_pair"][key]
            requested_actions[query["query_id"]] = query
    action_results = _query(model, list(requested_actions.values()), pad_token_id, 24)

    prepared_by_id = {row["trajectory_id"]: row for row in prepared["rows"]}
    output = []
    for trajectory_id in sorted(dataset_by_id):
        source_row = dataset_by_id[trajectory_id]
        prepared_row = prepared_by_id[trajectory_id]
        first_action, second_action = source_row["report_order"]
        first_result = first_results[prepared_row["first_report"]["query_id"]]
        second_query = prepared_row["second_report_by_first_choice"][
            first_result["legal_choice"]
        ]
        second_result = second_results[second_query["query_id"]]
        reports = {
            first_action: {
                "choice": first_result["legal_choice"],
                "expected": source_row["option_reports"][first_action]["expected"],
                "correct": first_result["legal_choice"]
                == source_row["option_reports"][first_action]["expected"],
                "result": first_result,
            },
            second_action: {
                "choice": second_result["legal_choice"],
                "expected": source_row["option_reports"][second_action]["expected"],
                "correct": second_result["legal_choice"]
                == source_row["option_reports"][second_action]["expected"],
                "result": second_result,
            },
        }
        arms = {}
        for arm, key in routing[trajectory_id].items():
            query = prepared_row["action_by_report_pair"][key]
            result = action_results[query["query_id"]]
            arms[arm] = {
                "selected_action": result["legal_choice"],
                "action_correct": result["legal_choice"] == source_row["expected_action"],
                "report_pair_key": key,
                "result": result,
            }
        output.append(
            {
                **source_row,
                "self_reports": reports,
                "trajectory_arms": arms,
            }
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    payload = {
        "schema_version": 1,
        "metadata": {
            "run_id": uuid.uuid4().hex,
            "created_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": elapsed,
            "source_dataset_sha256": dataset["content_sha256"],
            "prepared_payload_sha256": prepared["content_sha256"],
            "config_sha256": canonical_sha256(config),
            "model_id": MODEL_ID,
            "model_revision_requested": MODEL_REVISION,
            "model_revision_resolved": model_info(MODEL_ID, revision=MODEL_REVISION).sha,
            "dtype": "bfloat16",
            "thinking": False,
            "gpu_actual": torch.cuda.get_device_name(0),
            "torch_version": str(torch.__version__),
            "transformers_version": transformers.__version__,
            "modal_scope": "model_load_and_gpu_forward_passes_only",
        },
        "rows": output,
    }
    return json.dumps(payload, allow_nan=False, sort_keys=True)


@app.local_entrypoint()
def behavior() -> None:
    config = _load_json(CONFIG_PATH)
    dataset = _load_json(DATASET_PATH)
    prepared = _load_json(PREPARED_PATH)
    _validate(config, dataset, prepared)
    ledger = RESULT_ROOT / "cost_ledger.jsonl"
    estimate = estimate_cost(
        str(config["execution"]["gpu"]),
        float(config["execution"]["estimated_ceiling_seconds"]),
        cpu_cores=8,
        memory_gib=32,
    )
    admit_run(
        ledger,
        estimate,
        study_limit_usd=float(config["execution"]["hard_cost_limit_usd"]),
    )
    payload = json.loads(full_action_trajectory_gpu.remote())
    payload["metadata"]["git_commit"] = _git_head()
    target = RESULT_ROOT / "raw/behavior_v1.json"
    _write_new(target, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    measured = estimate_cost(
        str(config["execution"]["gpu"]),
        float(payload["metadata"]["elapsed_seconds"]),
        cpu_cores=8,
        memory_gib=32,
    )
    append_ledger(
        ledger,
        measured,
        run_id=str(payload["metadata"]["run_id"]),
        stage="full_action_trajectory_behavior_v1",
    )
    print(
        json.dumps(
            {
                "run_id": payload["metadata"]["run_id"],
                "rows": len(payload["rows"]),
                "elapsed_seconds": payload["metadata"]["elapsed_seconds"],
                "measured_cost_usd": measured.subtotal_usd,
                "buffered_cost_usd": measured.buffered_usd,
                "output": str(target),
            },
            indent=2,
            sort_keys=True,
        )
    )
