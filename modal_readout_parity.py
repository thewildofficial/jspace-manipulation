"""Thin real-checkpoint parity gate for the normalized candidate readout.

Runs ONLY the versioned scoring check against the pinned Qwen checkpoint and
pinned jlens adapter. No behavior, capture, patch, or trajectory work. No
frozen protocol, dataset, or historical artifact is touched::

    modal run modal_readout_parity.py::parity

Cost ceiling: 1200s on A100-80GB / 8 CPU / 32 GiB, reserved up front via
``jspace_policy.budget.estimate_cost``. Buffered ceiling is ~$1.25,
inside the $2 numerical-parity preflight allocation.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import modal

from jspace_policy.budget import estimate_cost

MODEL_ID = "Qwen/Qwen3.6-27B"
MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
JLENS_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_FILENAME = "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt"
LENS_REVISION = "qwen-n1000"
READOUT_VERSION = "normalized_candidates_v2"
RESULT_DIR = Path("results/research_audit")
TIMEOUT_SECONDS = 1200
PARITY_SEED = 82431
PROBE_SCALES = (0.1, 1.0, 10.0)
TOKEN_SETS = ((0, 1, 2), (100, 5000, 50000))

app = modal.App("jspace-readout-parity")
cache = modal.Volume.from_name("jspace-hf-cache", create_if_missing=True)
parity_image = (
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


@app.function(
    image=parity_image,
    gpu="A100-80GB",
    cpu=8,
    memory=32768,
    volumes={"/cache": cache},
    timeout=TIMEOUT_SECONDS,
    retries=0,
)
def parity_remote() -> str:
    import torch
    import transformers
    from huggingface_hub import model_info
    from jlens import JacobianLens, from_hf

    from jspace_policy.lens_readout import check_selected_unembed, selected_unembed

    started = time.perf_counter()
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
    if int(lens.d_model) != int(model.d_model):
        raise RuntimeError("pinned Jacobian Lens residual width mismatch")
    resolved = model_info(MODEL_ID, revision=MODEL_REVISION).sha
    checks = []
    with torch.inference_mode():
        for token_ids in TOKEN_SETS:
            generator = torch.Generator(device="cpu").manual_seed(PARITY_SEED)
            probe_state = torch.randn(
                (3, int(model.d_model)), generator=generator, dtype=torch.float32
            )
            probe_state *= torch.tensor(PROBE_SCALES)[:, None]
            states = probe_state.to("cuda")
            reference = model.unembed(states)[..., list(token_ids)]
            selected = selected_unembed(model, states, list(token_ids))
            max_abs_diff = float((selected - reference).abs().max())
            check_selected_unembed(model, states, list(token_ids))
            checks.append(
                {
                    "token_ids": list(token_ids),
                    "scales": list(PROBE_SCALES),
                    "seed": PARITY_SEED,
                    "max_abs_diff": max_abs_diff,
                    "status": "assert_close_passed",
                }
            )
    cache.commit()
    elapsed = time.perf_counter() - started
    return json.dumps(
        {
            "schema_version": 1,
            "readout_version": READOUT_VERSION,
            "status": "real_checkpoint_parity_passed",
            "created_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": elapsed,
            "model_id": MODEL_ID,
            "model_revision_requested": MODEL_REVISION,
            "model_revision_resolved": resolved,
            "lens_repo": LENS_REPO,
            "lens_filename": LENS_FILENAME,
            "lens_revision": LENS_REVISION,
            "lens_code_commit": JLENS_COMMIT,
            "gpu_actual": torch.cuda.get_device_name(0),
            "torch_version": str(torch.__version__),
            "transformers_version": transformers.__version__,
            "checks": checks,
        },
        sort_keys=True,
    )


@app.local_entrypoint()
def parity() -> None:
    ceiling = estimate_cost("A100-80GB", TIMEOUT_SECONDS, cpu_cores=8, memory_gib=32)
    print(
        json.dumps(
            {
                "reserved_ceiling_seconds": TIMEOUT_SECONDS,
                "reserved_subtotal_usd": round(ceiling.subtotal_usd, 5),
                "reserved_buffered_usd": round(ceiling.buffered_usd, 5),
            },
            sort_keys=True,
        )
    )
    payload = json.loads(parity_remote.remote())
    run_id = uuid.uuid4().hex[:8]
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULT_DIR / f"readout_parity_{run_id}.json"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"artifact": str(path), **payload}, indent=2, sort_keys=True))
