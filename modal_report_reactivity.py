"""Manually dispatched, discovery-only sprint scoring. CPU preparation is separate."""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

from jspace_policy.sprint_runtime import digest, reserve, verify_payload, write_new

app = modal.App("report-reactivity-discovery")
cache = modal.Volume.from_name("jspace-hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install("torch==2.13.0", "transformers==5.15.1", "huggingface_hub==1.30.0")
    .env({"HF_HOME": "/cache/huggingface", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir("src/jspace_policy", remote_path="/root/jspace_policy")
)
TIMEOUT = 900
# Rates verified September 5, 2026: https://modal.com/pricing . CPU is 8 physical cores.
CEILING_USD = TIMEOUT * (0.000694 + 8 * 0.0000131 + 32 * 0.00000222)


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=8,
    memory=32768,
    timeout=TIMEOUT,
    retries=0,
    volumes={"/cache": cache},
)
def score_gpu(payload: dict, batch_size: int = 8) -> dict:
    import torch
    import transformers

    verify_payload(payload)
    if not 1 <= batch_size <= 16:
        raise ValueError("batch size outside preflight range")
    started = time.perf_counter()
    torch.manual_seed(82431)
    model = (
        transformers.AutoModelForCausalLM.from_pretrained(
            payload["model_id"],
            revision=payload["revision"],
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        .cuda()
        .eval()
    )
    loaded = time.perf_counter()
    queries = sorted(payload["queries"].items(), key=lambda pair: (pair[1]["length"], pair[0]))

    def score(part):
        width = max(q["length"] for _, q in part)
        ids = torch.full(
            (len(part), width), payload["pad_token_id"], dtype=torch.long, device="cuda"
        )
        mask = torch.zeros_like(ids)
        for i, (_, query) in enumerate(part):
            ids[i, -query["length"] :] = torch.tensor(query["input_ids"], device="cuda")
            mask[i, -query["length"] :] = 1
        with torch.inference_mode():
            logits = (
                model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1)
                .logits[:, -1]
                .float()
            )
        result = {}
        for i, (key, query) in enumerate(part):
            values = logits[i, query["candidate_ids"]]
            result[key] = {
                "choice": query["labels"][int(values.argmax())],
                "logits": values.cpu().tolist(),
                "top1_token_id": int(logits[i].argmax()),
                "format_valid": int(logits[i].argmax()) in query["candidate_ids"],
            }
        return result

    # Identical full-prefix recomputation and batch/singleton parity are required
    # independently for each checkpoint; no hybrid-model cache cloning.
    probe = queries[: min(batch_size, len(queries))]
    first, replay = score(probe), score(probe)
    singleton = {key: score([(key, q)])[key] for key, q in probe}
    replay_error = max(
        abs(a - b)
        for key in first
        for a, b in zip(first[key]["logits"], replay[key]["logits"], strict=True)
    )
    batch_error = max(
        abs(a - b)
        for key in first
        for a, b in zip(first[key]["logits"], singleton[key]["logits"], strict=True)
    )
    choices_agree = all(first[key]["choice"] == singleton[key]["choice"] for key in first)
    parity = {
        "replay_max_abs": replay_error,
        "batch_single_max_abs": batch_error,
        "choices_agree": choices_agree,
        "passed": replay_error <= 0.01 and batch_error <= 0.25 and choices_agree,
    }
    results = {}
    if parity["passed"]:
        for start in range(0, len(queries), batch_size):
            results.update(score(queries[start : start + batch_size]))
    return {
        "payload_sha256": payload["sha256"],
        "parity": parity,
        "scores": results,
        "model_id": payload["model_id"],
        "revision": payload["revision"],
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "gpu": torch.cuda.get_device_name(),
        "model_class": type(model).__name__,
        "load_seconds": loaded - started,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "status": "engineering_pilot" if parity["passed"] else "instrument_gate_failed",
    }


@app.local_entrypoint()
def main(prepared: str, run_id: str, stage: str = "preflight", batch_size: int = 8):
    if stage not in {"preflight", "baseline", "incident", "replication"}:
        raise ValueError("only behavior stages are implemented")
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("invalid run ID")
    payload = json.loads(Path(prepared).read_text())
    verify_payload(payload)
    if stage == "replication" and payload["split"] != "locked":
        raise ValueError("replication stage requires a locked confirmation payload")
    output = Path("results/report_reactivity") / run_id
    output.mkdir(parents=True, exist_ok=False)
    write_new(
        output / "input_manifest.json",
        {
            "payload_sha256": payload["sha256"],
            "prepared_path": prepared,
            "code_sha256": digest(
                {
                    str(p): p.read_text()
                    for p in [Path(__file__), *Path("src/jspace_policy").glob("*.py")]
                }
            ),
            "ceiling_usd": CEILING_USD,
            "stage": stage,
        },
    )
    reserve(Path("results/report_reactivity/reservations.jsonl"), run_id, stage, CEILING_USD)
    try:
        result = score_gpu.remote(payload, batch_size)
        write_new(output / "raw.json", result)
    except Exception as exc:
        write_new(
            output / "failure.json",
            {"error": str(exc), "reservation_retained": True, "automatic_retry": False},
        )
        raise
    print(json.dumps({key: value for key, value in result.items() if key != "scores"}))
