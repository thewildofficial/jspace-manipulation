"""Modal entry points for H0R causal-instrument recovery."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import modal

MODEL_ID = "Qwen/Qwen3.6-27B"
MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
JLENS_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"

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
