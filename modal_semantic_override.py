"""V5-RBG-2 semantic action-outcome override screen."""

from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

from jspace_policy.budget import admit_run, append_ledger, estimate_cost
from modal_revealed_belief_games import (
    MODEL_ID,
    MODEL_REVISION,
    SYSTEM_PROMPT,
    _canonical_sha256,
    _git_head,
    _prepare_query,
    _query,
    _write_new,
    cache,
    image,
)

CONFIG_PATH = Path("configs/v5/semantic_override/experiment.json")
DATASET_PATH = Path("configs/v5/semantic_override/dataset.json")
RESULT_ROOT = Path("results/v5_semantic_override")
app = modal.App("jspace-v5-semantic-override")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate(config: dict[str, Any], dataset: dict[str, Any]) -> None:
    from jspace_policy.semantic_override_games import verify_dataset_payload

    if config["status"] != (
        "preregistered_after_rbg1_before_rbg2_dataset_or_model_execution"
    ):
        raise RuntimeError("RBG-2 protocol is not prospectively frozen")
    if config["model"]["id"] != MODEL_ID:
        raise RuntimeError("model ID changed")
    if config["model"]["revision"] != MODEL_REVISION:
        raise RuntimeError("model revision changed")
    if float(config["execution"]["hard_cost_limit_usd"]) > 6.0:
        raise RuntimeError("RBG-2 ceiling may not exceed USD 6")
    verify_dataset_payload(dataset, config)


def _action_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": row["prompt"]},
    ]


def _report_messages(row: dict[str, Any], action: str) -> tuple[list[dict[str, str]], str]:
    from jspace_policy.semantic_override_games import report_question

    question, expected = report_question(row, action)
    return (
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{row['scenario']}\n{question}"},
        ],
        expected,
    )


@app.function(image=image, cpu=2, memory=8192, volumes={"/cache": cache}, timeout=900)
def semantic_preflight_remote(dataset: dict[str, Any], config: dict[str, Any]) -> str:
    import transformers

    _validate(config, dataset)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION
    )
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    lengths = []
    queries = 0
    for row in dataset["rows"]:
        prepared = _prepare_query(
            tokenizer,
            f"{row['condition_id']}:action",
            _action_messages(row),
            ["A", "B"],
        )
        lengths.append(prepared["sequence_length"])
        queries += 1
        for action in ("A", "B"):
            messages, _ = _report_messages(row, action)
            prepared = _prepare_query(
                tokenizer,
                f"{row['condition_id']}:{action}",
                messages,
                ["X", "Y"],
            )
            lengths.append(prepared["sequence_length"])
            queries += 1
    return json.dumps(
        {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "source_dataset_sha256": dataset["content_sha256"],
            "queries_checked": queries,
            "minimum_tokens": min(lengths),
            "maximum_tokens": max(lengths),
            "status": "all_prompt_and_one_token_continuation_checks_passed",
        },
        sort_keys=True,
    )


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=8,
    memory=32768,
    volumes={"/cache": cache},
    timeout=2400,
)
def semantic_behavior_remote(dataset: dict[str, Any], config: dict[str, Any]) -> str:
    import torch
    import transformers
    from huggingface_hub import model_info

    _validate(config, dataset)
    started = time.perf_counter()
    torch.manual_seed(82427)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).cuda()
    model.eval()
    action_queries = [
        _prepare_query(
            tokenizer,
            row["condition_id"],
            _action_messages(row),
            ["A", "B"],
        )
        for row in dataset["rows"]
    ]
    actions = _query(model, tokenizer, action_queries, 24)
    report_queries = []
    expected_reports: dict[str, str] = {}
    for row in dataset["rows"]:
        for action in ("A", "B"):
            query_id = f"{row['condition_id']}:{action}"
            messages, expected = _report_messages(row, action)
            expected_reports[query_id] = expected
            report_queries.append(
                _prepare_query(tokenizer, query_id, messages, ["X", "Y"])
            )
    reports = _query(model, tokenizer, report_queries, 24)
    output = []
    for row in dataset["rows"]:
        condition_id = row["condition_id"]
        selected = actions[condition_id]["legal_choice"]
        option_reports = {}
        for action in ("A", "B"):
            query_id = f"{condition_id}:{action}"
            result = reports[query_id]
            option_reports[action] = {
                "choice": result["legal_choice"],
                "expected": expected_reports[query_id],
                "correct": result["legal_choice"] == expected_reports[query_id],
                "result": result,
            }
        output.append(
            {
                **row,
                "selected_action": selected,
                "action_correct": selected == row["expected_action"],
                "action_result": actions[condition_id],
                "option_reports": option_reports,
            }
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    cache.commit()
    payload = {
        "schema_version": 1,
        "metadata": {
            "run_id": uuid.uuid4().hex,
            "created_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": elapsed,
            "source_dataset_sha256": dataset["content_sha256"],
            "config_sha256": _canonical_sha256(config),
            "model_id": MODEL_ID,
            "model_revision_requested": MODEL_REVISION,
            "model_revision_resolved": model_info(
                MODEL_ID, revision=MODEL_REVISION
            ).sha,
            "dtype": "bfloat16",
            "thinking": False,
            "gpu_actual": torch.cuda.get_device_name(0),
            "torch_version": str(torch.__version__),
            "transformers_version": transformers.__version__,
        },
        "rows": sorted(output, key=lambda row: row["condition_id"]),
    }
    return json.dumps(payload, allow_nan=False, sort_keys=True)


@app.local_entrypoint()
def preflight() -> None:
    config = _load_json(CONFIG_PATH)
    dataset = _load_json(DATASET_PATH)
    _validate(config, dataset)
    payload = json.loads(semantic_preflight_remote.remote(dataset, config))
    target = RESULT_ROOT / "raw/preflight_v1.json"
    _write_new(target, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


@app.local_entrypoint()
def behavior() -> None:
    config = _load_json(CONFIG_PATH)
    dataset = _load_json(DATASET_PATH)
    _validate(config, dataset)
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
    payload = json.loads(semantic_behavior_remote.remote(dataset, config))
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
        stage="semantic_behavior_v1",
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
