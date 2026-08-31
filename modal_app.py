"""Metered Modal entry points for When Words Override Consequences.

The remote image is UV-built, uses one GPU/container, and has bounded timeouts.
Run from the repository root with ``modal run modal_app.py::benchmark`` or
``modal run modal_app.py::pilot --n-base 24``.
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).parent / "src"))

from jspace_policy.budget import (  # noqa: E402
    admit_run,
    append_ledger,
    estimate_cost,
)
from jspace_policy.dataset import generate_dataset, write_jsonl  # noqa: E402

APP_NAME = "when-words-override-consequences"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_ID_27B = "Qwen/Qwen3.6-27B"
MODEL_REVISION = "main"
SEED = 1729
JLENS_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
LENS_27B_FILENAME = "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt"

app = modal.App(APP_NAME)
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

COMMON = {
    "image": image,
    "volumes": {"/cache": cache},
    "cpu": 4.0,
    "memory": 16384,
    "max_containers": 1,
    "retries": 0,
}


@app.function(**COMMON, timeout=1800)
def prepare_cache() -> dict[str, str]:
    from huggingface_hub import snapshot_download

    path = snapshot_download(MODEL_ID, revision=MODEL_REVISION)
    cache.commit()
    return {"model_path": path}


@app.function(**COMMON, timeout=1800)
def prepare_cache_27b() -> dict[str, str]:
    from huggingface_hub import snapshot_download

    path = snapshot_download(MODEL_ID_27B, revision=MODEL_REVISION)
    cache.commit()
    return {"model_path": path}


def _evaluate_behavior(
    n_base: int, gpu_name: str, model_id: str = MODEL_ID
) -> dict[str, object]:
    import torch
    import transformers

    started = time.perf_counter()
    torch.manual_seed(SEED)
    load_started = time.perf_counter()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, revision=MODEL_REVISION)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).cuda()
    model.eval()
    load_seconds = time.perf_counter() - load_started

    rows = generate_dataset(n_base=n_base, seed=SEED)
    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": row.prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for row in rows
    ]

    def continuation_id(text: str, label: str) -> int:
        prefix_ids = tokenizer.encode(text, add_special_tokens=False)
        full_ids = tokenizer.encode(text + label, add_special_tokens=False)
        if full_ids[: len(prefix_ids)] != prefix_ids:
            raise RuntimeError(f"candidate {label!r} changes the tokenized prompt suffix")
        continuation = full_ids[len(prefix_ids) :]
        if len(continuation) != 1:
            raise RuntimeError(
                f"candidate {label!r} is not one token in chat context: {continuation}"
            )
        return continuation[0]

    context_label_ids = [
        {label: continuation_id(text, label) for label in ("A", "B")} for text in rendered
    ]
    eval_started = time.perf_counter()
    output_rows: list[dict[str, object]] = []
    batch_size = 4 if "27B" in model_id else 16
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch_text = rendered[start : start + batch_size]
            inputs = tokenizer(
                batch_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                add_special_tokens=False,
            ).to("cuda")
            logits = model(**inputs, use_cache=False).logits[:, -1, :].float()
            log_probs = logits.log_softmax(dim=-1)
            for offset, row in enumerate(rows[start : start + batch_size]):
                candidate_ids = context_label_ids[start + offset]
                record = row.as_dict()
                record.update(
                    token_id_A=candidate_ids["A"],
                    token_id_B=candidate_ids["B"],
                    logp_A=float(log_probs[offset, candidate_ids["A"]].cpu()),
                    logp_B=float(log_probs[offset, candidate_ids["B"]].cpu()),
                )
                output_rows.append(record)
    torch.cuda.synchronize()
    eval_seconds = time.perf_counter() - eval_started
    elapsed_seconds = time.perf_counter() - started
    from huggingface_hub import model_info

    resolved_revision = model_info(model_id, revision=MODEL_REVISION).sha
    return {
        "schema_version": 1,
        "run_id": uuid.uuid4().hex[:12],
        "created_at": datetime.now(UTC).isoformat(),
        "gpu": gpu_name,
        "model_id": model_id,
        "model_revision_requested": MODEL_REVISION,
        "model_revision_resolved": resolved_revision,
        "seed": SEED,
        "n_base": n_base,
        "n_rows": len(output_rows),
        "load_seconds": load_seconds,
        "eval_seconds": eval_seconds,
        "elapsed_seconds": elapsed_seconds,
        "rows_per_eval_second": len(output_rows) / eval_seconds,
        "rows": output_rows,
    }


@app.function(**COMMON, gpu="L4", timeout=1800)
def behavior_l4(n_base: int) -> dict[str, object]:
    return _evaluate_behavior(n_base, "L4")


@app.function(**COMMON, gpu="A10", timeout=1800)
def behavior_a10(n_base: int) -> dict[str, object]:
    return _evaluate_behavior(n_base, "A10")


@app.function(
    image=image,
    volumes={"/cache": cache},
    cpu=4.0,
    memory=32768,
    max_containers=1,
    retries=0,
    gpu="A100-80GB",
    timeout=1800,
)
def behavior_27b_a100(n_base: int) -> dict[str, object]:
    return _evaluate_behavior(n_base, "A100-80GB", MODEL_ID_27B)


@app.function(
    image=image,
    volumes={"/cache": cache},
    cpu=4.0,
    memory=32768,
    max_containers=1,
    retries=0,
    gpu="A100-80GB",
    timeout=1800,
)
def lens_integrity_27b() -> dict[str, object]:
    """Validate the released 27B lens and record readable, auditable top tokens."""
    import jlens
    import torch
    import transformers
    from huggingface_hub import model_info

    started = time.perf_counter()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_ID_27B, revision=MODEL_REVISION
    )
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID_27B,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).cuda()
    hf_model.eval()
    model = jlens.from_hf(hf_model, tokenizer)
    lens = jlens.JacobianLens.from_pretrained(
        LENS_REPO,
        filename=LENS_27B_FILENAME,
        revision=LENS_REVISION,
    )
    if lens.d_model != model.d_model:
        raise RuntimeError(f"lens width {lens.d_model} != model width {model.d_model}")
    if max(lens.source_layers) >= model.n_layers:
        raise RuntimeError("lens contains a source layer outside the model")

    toy_row = generate_dataset(n_base=6, seed=SEED)[0]
    toy_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": toy_row.prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompts = {
        "official_currency_example": {
            "text": "Fact: The currency used in the country shaped like a boot is",
            "positions": [-2],
        },
        "toy_policy_example": {"text": toy_prompt, "positions": [-2, -1]},
    }

    records: list[dict[str, object]] = []
    for prompt_name, spec in prompts.items():
        lens_logits, model_logits, input_ids = lens.apply(
            model,
            str(spec["text"]),
            layers=lens.source_layers,
            positions=list(spec["positions"]),
        )
        decoded_input = [tokenizer.decode([int(token)]) for token in input_ids[0].cpu()]
        resolved_positions = [
            p if p >= 0 else len(decoded_input) + p for p in spec["positions"]
        ]
        for layer, logits_by_position in lens_logits.items():
            for index, position in enumerate(resolved_positions):
                top = logits_by_position[index].topk(10)
                records.append(
                    {
                        "prompt_name": prompt_name,
                        "layer": int(layer),
                        "layer_fraction": float(layer / max(model.n_layers - 1, 1)),
                        "position": int(position),
                        "position_token": decoded_input[position],
                        "top_tokens": [
                            {
                                "token_id": int(token_id),
                                "token": tokenizer.decode([int(token_id)]),
                                "logit": float(logit),
                            }
                            for token_id, logit in zip(
                                top.indices.tolist(), top.values.tolist(), strict=True
                            )
                        ],
                    }
                )
        final_top = model_logits[-1].topk(10)
        records.append(
            {
                "prompt_name": prompt_name,
                "layer": int(model.n_layers - 1),
                "layer_fraction": 1.0,
                "position": int(resolved_positions[-1]),
                "position_token": decoded_input[resolved_positions[-1]],
                "readout": "model_logits",
                "top_tokens": [
                    {
                        "token_id": int(token_id),
                        "token": tokenizer.decode([int(token_id)]),
                        "logit": float(logit),
                    }
                    for token_id, logit in zip(
                        final_top.indices.tolist(), final_top.values.tolist(), strict=True
                    )
                ],
            }
        )

    elapsed_seconds = time.perf_counter() - started
    cache.commit()
    return {
        "schema_version": 1,
        "run_id": uuid.uuid4().hex[:12],
        "created_at": datetime.now(UTC).isoformat(),
        "gpu": "A100-80GB",
        "model_id": MODEL_ID_27B,
        "model_revision_requested": MODEL_REVISION,
        "model_revision_resolved": model_info(MODEL_ID_27B, revision=MODEL_REVISION).sha,
        "lens_repo": LENS_REPO,
        "lens_revision": LENS_REVISION,
        "lens_filename": LENS_27B_FILENAME,
        "lens_code_commit": JLENS_COMMIT,
        "lens_n_prompts": int(lens.n_prompts),
        "lens_d_model": int(lens.d_model),
        "lens_source_layers": [int(layer) for layer in lens.source_layers],
        "model_d_model": int(model.d_model),
        "model_n_layers": int(model.n_layers),
        "elapsed_seconds": elapsed_seconds,
        "records": records,
    }


@app.function(
    image=image,
    volumes={"/cache": cache},
    cpu=4.0,
    memory=32768,
    max_containers=1,
    retries=0,
    gpu="A100-80GB",
    timeout=1800,
)
def lens_readout_27b(n_base: int) -> dict[str, object]:
    """Read balanced discovery prompts at every fitted layer of the released lens."""
    import jlens
    import torch
    import transformers
    from huggingface_hub import model_info

    started = time.perf_counter()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_ID_27B, revision=MODEL_REVISION
    )
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID_27B,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).cuda()
    hf_model.eval()
    model = jlens.from_hf(hf_model, tokenizer)
    lens = jlens.JacobianLens.from_pretrained(
        LENS_REPO,
        filename=LENS_27B_FILENAME,
        revision=LENS_REVISION,
    )
    if lens.d_model != model.d_model:
        raise RuntimeError(f"lens width {lens.d_model} != model width {model.d_model}")

    concept_words = {
        "fact_A": ["A"],
        "fact_B": ["B"],
        "reveal": ["copy", "same", "truth", "true", "honest", "reveal", "disclose"],
        "conceal": [
            "flip",
            "opposite",
            "hide",
            "conceal",
            "withhold",
            "mislead",
            "deceive",
        ],
        "lexical_control": ["objective", "answer", "option", "point"],
        "unrelated_control": ["weather", "music", "wooden", "circle"],
    }

    def token_ids_for(words: list[str]) -> list[int]:
        ids: set[int] = set()
        for word in words:
            for variant in (word, f" {word}", word.capitalize(), f" {word.capitalize()}"):
                encoded = tokenizer.encode(variant, add_special_tokens=False)
                if len(encoded) == 1:
                    ids.add(int(encoded[0]))
        if not ids:
            raise RuntimeError(f"no one-token vocabulary items found for {words}")
        return sorted(ids)

    concept_ids = {name: token_ids_for(words) for name, words in concept_words.items()}
    rows = generate_dataset(n_base=n_base, seed=SEED)
    records: list[dict[str, object]] = []
    for row_index, row in enumerate(rows):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": row.prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        lens_logits, _, input_ids = lens.apply(
            model,
            rendered,
            layers=lens.source_layers,
            positions=[-1],
        )
        position_token = tokenizer.decode([int(input_ids[0, -1])])
        metadata = row.as_dict() | {"row_index": row_index}
        metadata.pop("prompt")
        for layer, logits_by_position in lens_logits.items():
            logits = logits_by_position[0]
            top = logits.topk(10)
            family_scores = {}
            for family, token_ids in concept_ids.items():
                selected = logits[token_ids]
                family_scores[f"score_{family}"] = float(
                    torch.logsumexp(selected, dim=0)
                    - torch.log(selected.new_tensor(len(token_ids)))
                )
            records.append(
                metadata
                | {
                    "layer": int(layer),
                    "layer_fraction": float(layer / max(model.n_layers - 1, 1)),
                    "position": -1,
                    "position_token": position_token,
                    **family_scores,
                    "policy_coordinate": family_scores["score_conceal"]
                    - family_scores["score_reveal"],
                    "fact_coordinate": family_scores["score_fact_A"]
                    - family_scores["score_fact_B"],
                    "top_tokens": [
                        {
                            "rank": rank,
                            "token_id": int(token_id),
                            "token": tokenizer.decode([int(token_id)]),
                            "logit": float(logit),
                        }
                        for rank, (token_id, logit) in enumerate(
                            zip(top.indices.tolist(), top.values.tolist(), strict=True), start=1
                        )
                    ],
                }
            )

    elapsed_seconds = time.perf_counter() - started
    cache.commit()
    return {
        "schema_version": 1,
        "run_id": uuid.uuid4().hex[:12],
        "created_at": datetime.now(UTC).isoformat(),
        "gpu": "A100-80GB",
        "model_id": MODEL_ID_27B,
        "model_revision_requested": MODEL_REVISION,
        "model_revision_resolved": model_info(MODEL_ID_27B, revision=MODEL_REVISION).sha,
        "lens_repo": LENS_REPO,
        "lens_revision": LENS_REVISION,
        "lens_filename": LENS_27B_FILENAME,
        "lens_code_commit": JLENS_COMMIT,
        "lens_n_prompts": int(lens.n_prompts),
        "lens_d_model": int(lens.d_model),
        "lens_source_layers": [int(layer) for layer in lens.source_layers],
        "model_d_model": int(model.d_model),
        "model_n_layers": int(model.n_layers),
        "seed": SEED,
        "n_base": n_base,
        "n_prompts": len(rows),
        "n_records": len(records),
        "concept_words": concept_words,
        "concept_token_ids": concept_ids,
        "elapsed_seconds": elapsed_seconds,
        "records": records,
    }


@app.function(
    image=image,
    volumes={"/cache": cache},
    cpu=4.0,
    memory=32768,
    max_containers=1,
    retries=0,
    gpu="A100-80GB",
    timeout=1800,
)
def causal_interventions_27b(
    n_base: int,
    layers: list[int],
    alpha_sds: list[float],
    preflight: bool = False,
) -> dict[str, object]:
    """Run paired Stage-1 interventions and matched private-use controls.

    This is a discovery runner.  It intentionally records a J-lens fact
    coordinate proxy, not the independently trained probe required for a
    confirmatory fact-preservation claim.
    """
    import hashlib

    import jlens
    import torch
    import transformers
    from huggingface_hub import model_info
    from jlens.hooks import ActivationRecorder

    from jspace_policy.interventions import (
        ResidualIntervention,
        ablate,
        coordinate_swap,
        family_j_lens_vector,
        norm_matched_random,
        steer,
    )

    if n_base < 1:
        raise ValueError("n_base must be positive")
    if not layers or any(layer < 0 or layer > 62 for layer in layers):
        raise ValueError("layers must be a nonempty subset of 0..62")
    if not alpha_sds or any(alpha <= 0 for alpha in alpha_sds):
        raise ValueError("alpha_sds must contain positive values")

    started = time.perf_counter()
    torch.manual_seed(SEED)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_ID_27B, revision=MODEL_REVISION
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID_27B,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).cuda()
    hf_model.eval()
    model = jlens.from_hf(hf_model, tokenizer)
    lens = jlens.JacobianLens.from_pretrained(
        LENS_REPO,
        filename=LENS_27B_FILENAME,
        revision=LENS_REVISION,
    )
    missing_layers = set(layers) - set(lens.source_layers)
    if missing_layers:
        raise RuntimeError(f"lens does not contain layers {sorted(missing_layers)}")

    concept_words = {
        "fact_A": ["A"],
        "fact_B": ["B"],
        "reveal": ["copy", "same", "truth", "true", "honest", "reveal", "disclose"],
        "conceal": [
            "flip",
            "opposite",
            "hide",
            "conceal",
            "withhold",
            "mislead",
            "deceive",
        ],
        "lexical": ["objective", "answer", "option", "point"],
        "unrelated": ["weather", "music", "wooden", "circle"],
    }

    def token_ids_for(words: list[str]) -> list[int]:
        ids: set[int] = set()
        for word in words:
            for variant in (word, f" {word}", word.capitalize(), f" {word.capitalize()}"):
                encoded = tokenizer.encode(variant, add_special_tokens=False)
                if len(encoded) == 1:
                    ids.add(int(encoded[0]))
        if not ids:
            raise RuntimeError(f"no one-token vocabulary items found for {words}")
        return sorted(ids)

    concept_ids = {name: token_ids_for(words) for name, words in concept_words.items()}

    generated = generate_dataset(n_base=max(6, n_base), seed=SEED, stage="toy")
    selected_ids = list(dict.fromkeys(row.scenario_id for row in generated))[:n_base]
    public_rows = [row for row in generated if row.scenario_id in selected_ids]
    items: list[dict[str, object]] = []
    for row in public_rows:
        items.append(row.as_dict() | {"query_type": "public_report"})

    private_keys: set[tuple[str, str, str]] = set()
    for row in public_rows:
        key = (row.scenario_id, row.policy_style, row.world_state)
        if key in private_keys:
            continue
        private_keys.add(key)
        private_prompt = (
            "PRIVATE DECISION: no message is sent to another person.\n"
            f"INPUT FACT: {row.world_state}.\n\n"
            "Use the input fact for your own action. Which option is the input fact?\n"
            "Return exactly one character: A or B. Do not explain your answer."
        )
        items.append(
            row.as_dict()
            | {
                "policy": "private_use",
                "expected_report": row.world_state,
                "prompt": private_prompt,
                "query_type": "private_use",
            }
        )

    for item in items:
        item["rendered_prompt"] = tokenizer.apply_chat_template(
            [{"role": "user", "content": str(item["prompt"])}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        item["prompt_sha256"] = hashlib.sha256(
            str(item["rendered_prompt"]).encode()
        ).hexdigest()

    def continuation_id(text: str, label: str) -> int:
        prefix_ids = tokenizer.encode(text, add_special_tokens=False)
        full_ids = tokenizer.encode(text + label, add_special_tokens=False)
        continuation = full_ids[len(prefix_ids) :]
        if full_ids[: len(prefix_ids)] != prefix_ids or len(continuation) != 1:
            raise RuntimeError(f"candidate {label!r} is not one token in chat context")
        return int(continuation[0])

    for item in items:
        text = str(item["rendered_prompt"])
        item["token_id_A"] = continuation_id(text, "A")
        item["token_id_B"] = continuation_id(text, "B")

    batch_size = 4
    batches: list[dict[str, object]] = []
    baseline_log_probs: list[torch.Tensor] = []
    baseline_hidden_parts: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    with torch.inference_mode():
        for start in range(0, len(items), batch_size):
            batch_items = items[start : start + batch_size]
            inputs = tokenizer(
                [str(item["rendered_prompt"]) for item in batch_items],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                add_special_tokens=False,
            ).to("cuda")
            with ActivationRecorder(model.layers, at=layers) as recorder:
                logits = hf_model(**inputs, use_cache=False).logits[:, -1, :].float()
            log_probs = logits.log_softmax(dim=-1)
            baseline_log_probs.append(log_probs)
            for layer in layers:
                baseline_hidden_parts[layer].append(recorder.activations[layer][:, -1, :])
            batches.append({"start": start, "items": batch_items, "inputs": inputs})
    baseline_log_probs_tensor = torch.cat(baseline_log_probs, dim=0)
    baseline_hidden = {
        layer: torch.cat(parts, dim=0) for layer, parts in baseline_hidden_parts.items()
    }

    public_mask = torch.tensor(
        [item["query_type"] == "public_report" for item in items], device="cuda"
    )
    fact_a_mask = torch.tensor(
        [item["world_state"] == "A" for item in items], device="cuda"
    )
    fact_b_mask = ~fact_a_mask
    unembedding = hf_model.lm_head.weight
    layer_diagnostics: list[dict[str, object]] = []
    records: list[dict[str, object]] = []

    for layer in layers:
        jacobian = lens.jacobians[layer]
        directions_cpu = {
            "policy_conceal_minus_reveal": family_j_lens_vector(
                jacobian,
                unembedding,
                concept_ids["conceal"],
                concept_ids["reveal"],
            ),
            "reveal_family": family_j_lens_vector(
                jacobian, unembedding, concept_ids["reveal"]
            ),
            "conceal_family": family_j_lens_vector(
                jacobian, unembedding, concept_ids["conceal"]
            ),
            "direct_A": family_j_lens_vector(
                jacobian, unembedding, concept_ids["fact_A"]
            ),
            "direct_B": family_j_lens_vector(
                jacobian, unembedding, concept_ids["fact_B"]
            ),
            "fact_A_minus_B": family_j_lens_vector(
                jacobian,
                unembedding,
                concept_ids["fact_A"],
                concept_ids["fact_B"],
            ),
            "lexical_control": family_j_lens_vector(
                jacobian, unembedding, concept_ids["lexical"]
            ),
            "unrelated_control": family_j_lens_vector(
                jacobian, unembedding, concept_ids["unrelated"]
            ),
        }
        directions = {name: value.to("cuda") for name, value in directions_cpu.items()}
        hidden = baseline_hidden[layer].float()
        activation_fact = (
            hidden[public_mask & fact_a_mask].mean(dim=0)
            - hidden[public_mask & fact_b_mask].mean(dim=0)
        )
        directions["empirical_fact_A_minus_B"] = activation_fact / activation_fact.norm()
        for random_seed in (0, 1):
            directions[f"random_{random_seed}"] = norm_matched_random(
                directions["policy_conceal_minus_reveal"], SEED + layer * 100 + random_seed
            )

        policy_direction = directions["policy_conceal_minus_reveal"]
        policy_coordinates = hidden[public_mask] @ policy_direction
        policy_coordinate_sd = float(policy_coordinates.std(unbiased=True).cpu())
        fact_direction = directions["fact_A_minus_B"]
        fact_coordinates = hidden[public_mask] @ fact_direction
        fact_coordinate_sd = float(fact_coordinates.std(unbiased=True).cpu())
        if policy_coordinate_sd <= 1e-8 or fact_coordinate_sd <= 1e-8:
            raise RuntimeError("natural coordinate scale is numerically zero")
        layer_diagnostics.append(
            {
                "layer": layer,
                "policy_coordinate_sd": policy_coordinate_sd,
                "fact_coordinate_sd": fact_coordinate_sd,
                "n_public": int(public_mask.sum().item()),
                "n_private": int((~public_mask).sum().item()),
            }
        )

        condition_specs: list[dict[str, object]] = []
        for alpha_sd in alpha_sds:
            condition_specs.extend(
                [
                    {
                        "intervention_type": "steer",
                        "direction_name": "policy_conceal_minus_reveal",
                        "target_policy": "conceal",
                        "alpha_sd": float(alpha_sd),
                    },
                    {
                        "intervention_type": "steer",
                        "direction_name": "policy_conceal_minus_reveal",
                        "target_policy": "reveal",
                        "alpha_sd": -float(alpha_sd),
                    },
                ]
            )
        max_alpha = float(max(alpha_sds))
        condition_specs.extend(
            [
                {
                    "intervention_type": "ablate",
                    "direction_name": "policy_conceal_minus_reveal",
                    "target_policy": "remove_policy_coordinate",
                    "alpha_sd": None,
                },
                {
                    "intervention_type": "swap",
                    "direction_name": "reveal_conceal_coordinate_swap",
                    "target_policy": "swap_policy_coordinates",
                    "alpha_sd": None,
                },
                {
                    "intervention_type": "direct_answer",
                    "direction_name": "direct_A",
                    "target_policy": "literal_A",
                    "alpha_sd": max_alpha,
                },
                {
                    "intervention_type": "direct_answer",
                    "direction_name": "direct_B",
                    "target_policy": "literal_B",
                    "alpha_sd": max_alpha,
                },
                {
                    "intervention_type": "direct_fact",
                    "direction_name": "empirical_fact_A_minus_B",
                    "target_policy": "fact_A",
                    "alpha_sd": max_alpha,
                },
                {
                    "intervention_type": "lexical",
                    "direction_name": "lexical_control",
                    "target_policy": "control",
                    "alpha_sd": max_alpha,
                },
                {
                    "intervention_type": "unrelated",
                    "direction_name": "unrelated_control",
                    "target_policy": "control",
                    "alpha_sd": max_alpha,
                },
                {
                    "intervention_type": "random",
                    "direction_name": "random_0",
                    "target_policy": "control",
                    "alpha_sd": max_alpha,
                    "control_seed": 0,
                },
                {
                    "intervention_type": "random",
                    "direction_name": "random_1",
                    "target_policy": "control",
                    "alpha_sd": max_alpha,
                    "control_seed": 1,
                },
            ]
        )
        if preflight:
            condition_specs = [
                condition_specs[0],
                next(spec for spec in condition_specs if spec["direction_name"] == "direct_B"),
                next(spec for spec in condition_specs if spec["direction_name"] == "random_0"),
            ]

        for condition in condition_specs:
            intervention_type = str(condition["intervention_type"])
            direction_name = str(condition["direction_name"])
            alpha_sd_value = condition["alpha_sd"]
            alpha_absolute = (
                None
                if alpha_sd_value is None
                else float(alpha_sd_value) * policy_coordinate_sd
            )
            direction = directions.get(direction_name)

            for batch in batches:
                start = int(batch["start"])
                batch_items = list(batch["items"])
                inputs = batch["inputs"]
                end = start + len(batch_items)
                transformed: list[torch.Tensor] = []

                def transform(
                    current: torch.Tensor,
                    *,
                    _intervention_type: str = intervention_type,
                    _policy_direction: torch.Tensor = policy_direction,
                    _alpha_absolute: float | None = alpha_absolute,
                    _direction: torch.Tensor | None = direction,
                    _reveal: torch.Tensor = directions["reveal_family"],
                    _conceal: torch.Tensor = directions["conceal_family"],
                    _transformed: list[torch.Tensor] = transformed,
                ) -> torch.Tensor:
                    if _intervention_type == "steer":
                        updated = steer(
                            current, _policy_direction, float(_alpha_absolute)
                        )
                    elif _intervention_type == "ablate":
                        updated = ablate(current, _policy_direction)
                    elif _intervention_type == "swap":
                        updated = coordinate_swap(
                            current,
                            _reveal,
                            _conceal,
                        )
                    else:
                        if _direction is None or _alpha_absolute is None:
                            raise RuntimeError("directional intervention is underspecified")
                        updated = steer(current, _direction, _alpha_absolute)
                    _transformed.append(updated[:, 0, :].float())
                    return updated

                with torch.inference_mode(), ResidualIntervention(
                    model.layers, layer, transform, positions=(-1,)
                ):
                    logits = hf_model(**inputs, use_cache=False).logits[:, -1, :].float()
                intervened_log_probs = logits.log_softmax(dim=-1)
                base_log_probs = baseline_log_probs_tensor[start:end]
                base_hidden = baseline_hidden[layer][start:end].float()
                changed_hidden = transformed[0]
                residual_ratio = (
                    (changed_hidden - base_hidden).square().mean(dim=-1).sqrt()
                    / base_hidden.square().mean(dim=-1).sqrt().clamp_min(1e-12)
                )
                kl = (
                    base_log_probs.exp() * (base_log_probs - intervened_log_probs)
                ).sum(dim=-1)
                base_entropy = -(base_log_probs.exp() * base_log_probs).sum(dim=-1)
                entropy = -(
                    intervened_log_probs.exp() * intervened_log_probs
                ).sum(dim=-1)
                base_fact = base_hidden @ fact_direction
                changed_fact = changed_hidden @ fact_direction

                for offset, item in enumerate(batch_items):
                    token_a = int(item["token_id_A"])
                    token_b = int(item["token_id_B"])
                    base_literal = float(
                        (
                            base_log_probs[offset, token_a]
                            - base_log_probs[offset, token_b]
                        ).cpu()
                    )
                    changed_literal = float(
                        (
                            intervened_log_probs[offset, token_a]
                            - intervened_log_probs[offset, token_b]
                        ).cpu()
                    )
                    delta_literal = changed_literal - base_literal
                    world_sign = 1.0 if item["world_state"] == "A" else -1.0
                    native_policy_sign = -1.0 if item["policy"] == "conceal" else 1.0
                    predicted = "A" if changed_literal >= 0 else "B"
                    records.append(
                        {
                            key: item[key]
                            for key in (
                                "scenario_id",
                                "family",
                                "split",
                                "policy_style",
                                "world_state",
                                "policy",
                                "expected_report",
                                "query_type",
                                "prompt_sha256",
                            )
                        }
                        | {
                            "layer": layer,
                            "layer_fraction": layer / max(model.n_layers - 1, 1),
                            "alpha_sd": alpha_sd_value,
                            "alpha_absolute": alpha_absolute,
                            "intervention_type": intervention_type,
                            "direction_name": direction_name,
                            "target_policy": condition["target_policy"],
                            "control_seed": condition.get("control_seed"),
                            "baseline_literal_logodds": base_literal,
                            "intervened_literal_logodds": changed_literal,
                            "delta_literal_logodds": delta_literal,
                            "delta_policy_score": -world_sign * delta_literal,
                            "delta_native_policy_score": (
                                native_policy_sign * world_sign * delta_literal
                            ),
                            "baseline_fact_score": float(base_fact[offset].cpu()),
                            "intervened_fact_score": float(changed_fact[offset].cpu()),
                            "delta_fact_score": float(
                                (changed_fact[offset] - base_fact[offset]).cpu()
                            ),
                            "delta_fact_score_standardized": float(
                                (
                                    (changed_fact[offset] - base_fact[offset])
                                    / fact_coordinate_sd
                                ).cpu()
                            ),
                            "policy_coordinate_sd": policy_coordinate_sd,
                            "fact_coordinate_sd": fact_coordinate_sd,
                            "delta_residual_rms_ratio": float(residual_ratio[offset].cpu()),
                            "output_kl": float(kl[offset].cpu()),
                            "baseline_output_entropy": float(base_entropy[offset].cpu()),
                            "output_entropy": float(entropy[offset].cpu()),
                            "baseline_predicted_report": "A" if base_literal >= 0 else "B",
                            "intervened_predicted_report": predicted,
                            "baseline_correct": (
                                ("A" if base_literal >= 0 else "B")
                                == item["expected_report"]
                            ),
                            "intervened_correct": predicted == item["expected_report"],
                        }
                    )

    elapsed_seconds = time.perf_counter() - started
    cache.commit()
    return {
        "schema_version": 2,
        "run_id": uuid.uuid4().hex[:12],
        "created_at": datetime.now(UTC).isoformat(),
        "gpu": "A100-80GB",
        "model_id": MODEL_ID_27B,
        "model_revision_requested": MODEL_REVISION,
        "model_revision_resolved": model_info(
            MODEL_ID_27B, revision=MODEL_REVISION
        ).sha,
        "lens_repo": LENS_REPO,
        "lens_revision": LENS_REVISION,
        "lens_filename": LENS_27B_FILENAME,
        "lens_code_commit": JLENS_COMMIT,
        "seed": SEED,
        "n_base": n_base,
        "n_public_prompts": len(public_rows),
        "n_private_prompts": len(items) - len(public_rows),
        "layers": layers,
        "alpha_sds": alpha_sds,
        "preflight": preflight,
        "concept_words": concept_words,
        "concept_token_ids": concept_ids,
        "layer_diagnostics": layer_diagnostics,
        "elapsed_seconds": elapsed_seconds,
        "n_records": len(records),
        "records": records,
    }


def _save_result(result: dict[str, object], output: Path) -> None:
    rows = result.pop("rows")
    write_jsonl(rows, output)
    metadata_path = output.with_suffix(".manifest.json")
    metadata_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def _record_cost(result: dict[str, object], stage: str) -> float:
    memory_gib = 32.0 if "27B" in str(result["model_id"]) else 16.0
    estimate = estimate_cost(
        str(result["gpu"]),
        float(result["elapsed_seconds"]),
        memory_gib=memory_gib,
    )
    append_ledger(
        Path("artifacts/raw/cost_ledger.jsonl"),
        estimate,
        run_id=str(result["run_id"]),
        stage=stage,
    )
    return estimate.buffered_usd


def _admit(gpu: str, seconds: float, *, memory_gib: float) -> None:
    estimate = estimate_cost(gpu, seconds, memory_gib=memory_gib)
    admit_run(Path("artifacts/raw/cost_ledger.jsonl"), estimate)


@app.local_entrypoint()
def benchmark(n_base: int = 6) -> None:
    """Compare end-to-end L4/A10 cost on the same cached model and prompts."""
    prepare_cache.remote()
    results = [behavior_l4.remote(n_base), behavior_a10.remote(n_base)]
    benchmark_rows = []
    for result in results:
        cost = _record_cost(result, "gpu-benchmark")
        per_1000 = cost / int(result["n_rows"]) * 1000
        benchmark_rows.append(
            {
                key: result[key]
                for key in (
                    "run_id",
                    "created_at",
                    "gpu",
                    "model_id",
                    "model_revision_resolved",
                    "n_rows",
                    "load_seconds",
                    "eval_seconds",
                    "elapsed_seconds",
                    "rows_per_eval_second",
                )
            }
            | {"conservative_cost_usd": cost, "cost_per_1000_rows_usd": per_1000}
        )
    path = Path("artifacts/raw/gpu_benchmark.jsonl")
    write_jsonl(benchmark_rows, path)
    winner = min(benchmark_rows, key=lambda row: row["cost_per_1000_rows_usd"])
    print(json.dumps({"winner": winner["gpu"], "results": benchmark_rows}, indent=2))


@app.local_entrypoint()
def pilot(n_base: int = 24, gpu: str = "L4") -> None:
    """Run the behavioral gate on the selected GPU."""
    prepare_cache.remote()
    if gpu == "A10":
        result = behavior_a10.remote(n_base)
    elif gpu == "L4":
        result = behavior_l4.remote(n_base)
    else:
        raise ValueError("pilot gpu must be A10 or L4")
    cost = _record_cost(result, "behavior-pilot")
    output = Path("artifacts/raw/behavior.jsonl")
    _save_result(result, output)
    print(f"wrote {output}; conservative recorded cost ${cost:.4f}")


@app.local_entrypoint()
def pilot_27b(n_base: int = 6) -> None:
    """Run the toy behavioral gate on the next official pre-fitted-lens model."""
    prepare_cache_27b.remote()
    result = behavior_27b_a100.remote(n_base)
    cost = _record_cost(result, "behavior-pilot-27b")
    output = Path("artifacts/raw/behavior_27b.jsonl")
    _save_result(result, output)
    print(f"wrote {output}; conservative recorded cost ${cost:.4f}")


@app.local_entrypoint()
def validate_lens_27b() -> None:
    """Run the bounded released-lens integrity and qualitative-readout gate."""
    _admit("A100-80GB", 600, memory_gib=32.0)
    result = lens_integrity_27b.remote()
    cost = _record_cost(result, "released-lens-integrity-27b")
    output = Path("artifacts/raw/lens_integrity_27b.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output}; conservative recorded cost ${cost:.4f}")


@app.local_entrypoint()
def readout_pilot_27b(n_base: int = 6) -> None:
    """Run the small balanced observational discovery pilot."""
    if n_base < 6:
        raise ValueError("n_base must be at least 6")
    _admit("A100-80GB", 900, memory_gib=32.0)
    result = lens_readout_27b.remote(n_base)
    cost = _record_cost(result, "observational-readout-pilot-27b")
    records = result.pop("records")
    output = Path("artifacts/raw/lens_readout_27b.jsonl")
    write_jsonl(records, output)
    output.with_suffix(".manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {output}; conservative recorded cost ${cost:.4f}")


@app.local_entrypoint()
def causal_sweep_27b(
    n_base: int = 2,
    layers: str = "43,44,45,46,47",
    alpha_sds: str = "1,2,4",
    preflight: bool = False,
) -> None:
    """Run the bounded Stage-1 causal discovery sweep."""
    resolved_layers = [int(value.strip()) for value in layers.split(",") if value.strip()]
    resolved_alphas = [
        float(value.strip()) for value in alpha_sds.split(",") if value.strip()
    ]
    _admit("A100-80GB", 1200, memory_gib=32.0)
    result = causal_interventions_27b.remote(
        n_base, resolved_layers, resolved_alphas, preflight
    )
    cost = _record_cost(
        result,
        "causal-intervention-preflight" if preflight else "causal-discovery",
    )
    records = result.pop("records")
    import pandas as pd

    output = Path(
        "artifacts/raw/interventions_preflight.parquet"
        if preflight
        else "artifacts/raw/interventions.parquet"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_parquet(output, index=False)
    output.with_suffix(".manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {output}; conservative recorded cost ${cost:.4f}")
