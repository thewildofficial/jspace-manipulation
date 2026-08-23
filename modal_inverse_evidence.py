"""V5-RBG-4 inverse-evidence screen."""

from __future__ import annotations

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

CONFIG_PATH = Path("configs/v5/inverse_evidence/experiment.json")
DATASET_PATH = Path("configs/v5/inverse_evidence/dataset.json")
RESULT_ROOT = Path("results/v5_inverse_evidence")
MODEL_ID = "Qwen/Qwen3.6-27B"
MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
SYSTEM_PROMPT = (
    "Treat the described game and prior episodes as exact formal information. "
    "Follow the requested forced-choice format exactly."
)

app = modal.App("jspace-v5-inverse-evidence")
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
)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_new(path: Path, value: str) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite prospective artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _continuation_id(tokenizer: Any, rendered: str, answer: str) -> int:
    prefix = tokenizer.encode(rendered, add_special_tokens=False)
    full = tokenizer.encode(rendered + answer, add_special_tokens=False)
    if full[: len(prefix)] == prefix and len(full) == len(prefix) + 1:
        return int(full[-1])
    raise ValueError(f"{answer!r} is not one token after the frozen Answer: prefix")


def _prepare_query(
    tokenizer: Any,
    query_id: str,
    messages: list[dict[str, str]],
    candidates: list[str],
) -> dict[str, Any]:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    token_ids = list(map(int, tokenizer.encode(rendered, add_special_tokens=False)))
    return {
        "query_id": query_id,
        "prompt_token_ids": token_ids,
        "candidate_labels": candidates,
        "candidate_token_ids": [
            _continuation_id(tokenizer, rendered, candidate) for candidate in candidates
        ],
        "sequence_length": len(token_ids),
    }


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
    tokenizer: Any,
    rows: list[dict[str, Any]],
    batch_size: int,
) -> dict[str, dict[str, Any]]:
    import torch

    output: dict[str, dict[str, Any]] = {}
    ordered = sorted(rows, key=lambda row: (row["sequence_length"], row["query_id"]))
    for start in range(0, len(ordered), batch_size):
        part = ordered[start : start + batch_size]
        input_ids, attention = _left_padded(part, tokenizer.pad_token_id)
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
                "top1_text": tokenizer.decode([top1_id]),
                "formatting_compliant": top1_id in candidate_ids,
            }
        del input_ids, attention, logits
    return output


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate(config: dict[str, Any], dataset: dict[str, Any]) -> None:
    from jspace_policy.inverse_evidence_games import verify_dataset_payload

    if config["status"] != (
        "preregistered_after_rbg3_before_rbg4_dataset_or_model_execution"
    ):
        raise RuntimeError("RBG-4 protocol is not prospectively frozen")
    if config["model"]["id"] != MODEL_ID:
        raise RuntimeError("model ID changed")
    if config["model"]["revision"] != MODEL_REVISION:
        raise RuntimeError("model revision changed")
    if float(config["execution"]["hard_cost_limit_usd"]) > 6.0:
        raise RuntimeError("RBG-4 ceiling may not exceed USD 6")
    verify_dataset_payload(dataset, config)


def _action_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": row["prompt"]},
    ]


def _report_messages(row: dict[str, Any], action: str) -> tuple[list[dict[str, str]], str]:
    from jspace_policy.inverse_evidence_games import report_question

    question, expected = report_question(row, action)
    return (
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{row['scenario']}\n{question}"},
        ],
        expected,
    )


@app.function(image=image, cpu=2, memory=8192, volumes={"/cache": cache}, timeout=900)
def inverse_evidence_preflight_remote(
    dataset: dict[str, Any], config: dict[str, Any]
) -> str:
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
def inverse_evidence_behavior_remote(
    dataset: dict[str, Any], config: dict[str, Any]
) -> str:
    import torch
    import transformers
    from huggingface_hub import model_info

    _validate(config, dataset)
    started = time.perf_counter()
    torch.manual_seed(82429)
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
    payload = json.loads(inverse_evidence_preflight_remote.remote(dataset, config))
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
    payload = json.loads(inverse_evidence_behavior_remote.remote(dataset, config))
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
        stage="inverse_evidence_behavior_v1",
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
