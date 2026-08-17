"""Gated execution for V2-E1: Strategic Workspace Atlas.

Commands:
    modal run modal_workspace_atlas.py::freeze_dataset
    modal run modal_workspace_atlas.py::behavior --phase open
    modal run modal_workspace_atlas.py::mechanistic --phase open
    modal run modal_workspace_atlas.py::behavior --phase locked
    modal run modal_workspace_atlas.py::mechanistic --phase locked
"""

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

MODEL_ID = "Qwen/Qwen3.6-27B"
MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
JLENS_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
LENS_FILENAME = "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt"

app = modal.App("jspace-v2-e1-workspace-atlas")
cache = modal.Volume.from_name("jspace-hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .uv_pip_install(
        "numpy>=2.0",
        "scikit-learn>=1.6",
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


def _content_hash_valid(payload: dict[str, Any]) -> bool:
    claimed = payload.get("content_sha256")
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    return claimed == _canonical_sha256(body)


def _continuation_id(tokenizer: Any, prompt: str, answer: str) -> int:
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    full = tokenizer.encode(prompt + answer, add_special_tokens=False)
    if full[: len(prefix)] == prefix and len(full) == len(prefix) + 1:
        return int(full[-1])
    raise ValueError(f"{answer!r} is not one token after the frozen Answer: prefix")


def _tokenize_payload(config: dict[str, Any]) -> dict[str, Any]:
    import transformers

    from jspace_policy.workspace_atlas import dataset_payload, verify_dataset_payload

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    payload = dataset_payload(config)
    verify_dataset_payload(payload)
    for row in payload["rows"]:
        task_text = str(row["prompt"])
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": task_text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        row["task_text"] = task_text
        row["task_sha256"] = hashlib.sha256(task_text.encode()).hexdigest()
        row["prompt"] = prompt
        row["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
        row["prompt_token_ids"] = tokenizer.encode(prompt, add_special_tokens=False)
        row["candidate_token_ids"] = [
            _continuation_id(tokenizer, prompt, candidate) for candidate in row["candidates"]
        ]
        if len(set(row["candidate_token_ids"])) != len(row["candidates"]):
            raise RuntimeError(f"candidate token collision in {row['condition_id']}")
        answer_index = row["candidates"].index(row["expected_action"])
        row["expected_action_token_id"] = row["candidate_token_ids"][answer_index]
        row["sequence_length"] = len(row["prompt_token_ids"])
    payload["status"] = "tokenized_and_frozen_before_behavior"
    payload["tokenizer_revision"] = MODEL_REVISION
    payload["created_at"] = datetime.now(UTC).isoformat()
    payload["content_sha256"] = _canonical_sha256(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )
    cache.commit()
    return payload


@app.function(image=image, volumes={"/cache": cache}, cpu=2.0, memory=4096, timeout=900)
def tokenize_dataset_remote(config: dict[str, Any]) -> str:
    return json.dumps(_tokenize_payload(config), sort_keys=True)


def _load_model(*, with_lens: bool) -> tuple[Any, Any | None, dict[str, Any]]:
    import torch
    import transformers
    from huggingface_hub import model_info
    from jlens import JacobianLens, from_hf

    seed = 77891
    torch.manual_seed(seed)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
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
        if lens.d_model != model.d_model:
            raise RuntimeError(f"lens width {lens.d_model} != model width {model.d_model}")
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
        "seed": seed,
        "dtype": "bfloat16",
        "gpu_requested": "A100-80GB",
        "gpu_actual": torch.cuda.get_device_name(0),
        "torch_version": str(torch.__version__),
        "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda,
    }
    return model, lens, metadata


def _select_rows(
    dataset: dict[str, Any], config: dict[str, Any], phase: str
) -> list[dict[str, Any]]:
    if phase == "open":
        splits = set(config["execution"]["open_splits"])
    elif phase == "locked":
        splits = {config["execution"]["locked_split"]}
    else:
        raise ValueError("phase must be open or locked")
    return [row for row in dataset["rows"] if row["split"] in splits]


def _left_padded_batch(rows: list[dict[str, Any]]) -> tuple[Any, Any]:
    import torch

    width = max(len(row["prompt_token_ids"]) for row in rows)
    input_ids = torch.zeros((len(rows), width), dtype=torch.long, device="cuda")
    attention = torch.zeros_like(input_ids)
    for index, row in enumerate(rows):
        tokens = torch.tensor(row["prompt_token_ids"], dtype=torch.long, device="cuda")
        input_ids[index, width - len(tokens) :] = tokens
        attention[index, width - len(tokens) :] = 1
    return input_ids, attention


def _behavior_rows(
    model: Any, rows: list[dict[str, Any]], batch_size: int
) -> list[dict[str, Any]]:
    import torch

    output: list[dict[str, Any]] = []
    ordered = sorted(rows, key=lambda row: (row["sequence_length"], row["condition_id"]))
    for start in range(0, len(ordered), batch_size):
        part = ordered[start : start + batch_size]
        input_ids, attention = _left_padded_batch(part)
        with torch.inference_mode():
            logits = model._hf_model(
                input_ids, attention_mask=attention, logits_to_keep=1
            ).logits
        for index, row in enumerate(part):
            final = logits[index, -1].float()
            log_probs = final.log_softmax(-1)
            candidate_ids = [int(token) for token in row["candidate_token_ids"]]
            legal_logits = final[candidate_ids]
            legal_log_probs = legal_logits.log_softmax(-1)
            top1 = int(final.argmax().detach().cpu())
            legal_choice = int(legal_logits.argmax().detach().cpu())
            output.append(
                {
                    "condition_id": row["condition_id"],
                    "matched_group_id": row["matched_group_id"],
                    "game": row["game"],
                    "split": row["split"],
                    "expected_action": row["expected_action"],
                    "expected_action_token_id": row["expected_action_token_id"],
                    "top1_token_id": top1,
                    "top1_text": model.tokenizer.decode([top1]),
                    "correct": top1 == int(row["expected_action_token_id"]),
                    "formatting_compliant": top1 in candidate_ids,
                    "legal_choice": row["candidates"][legal_choice],
                    "legal_choice_correct": row["candidates"][legal_choice]
                    == row["expected_action"],
                    "candidate_log_probs": [
                        float(log_probs[token].cpu()) for token in candidate_ids
                    ],
                    "legal_action_log_probs": [float(value.cpu()) for value in legal_log_probs],
                }
            )
        del input_ids, attention, logits
    return sorted(output, key=lambda row: row["condition_id"])


def _behavior_summary(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    exact = set(config["behavior"]["exact_games"])
    exact_rows = [row for row in rows if row["game"] in exact]
    formatting = sum(row["formatting_compliant"] for row in rows) / len(rows)
    exact_accuracy = sum(row["correct"] for row in exact_rows) / len(exact_rows)
    family = {
        game: sum(row["correct"] for row in rows if row["game"] == game)
        / sum(row["game"] == game for row in rows)
        for game in sorted({row["game"] for row in rows})
    }
    exact_min = min(family[game] for game in exact)
    gate = (
        formatting >= float(config["behavior"]["minimum_formatting_compliance"])
        and exact_accuracy >= float(config["behavior"]["minimum_exact_game_optimal_accuracy"])
        and exact_min >= float(config["behavior"]["minimum_exact_game_family_accuracy"])
    )
    return {
        "n_rows": len(rows),
        "formatting_compliance": formatting,
        "exact_game_optimal_accuracy": exact_accuracy,
        "exact_game_minimum_family_accuracy": exact_min,
        "family_accuracy": family,
        "gate_pass": gate,
    }


@app.function(
    image=image,
    volumes={"/cache": cache},
    cpu=4.0,
    memory=32768,
    gpu="A100-80GB",
    timeout=3600,
    max_containers=1,
    retries=0,
)
def behavior_remote(
    phase: str, dataset: dict[str, Any], config: dict[str, Any], git_commit: str
) -> str:
    import torch

    if not _content_hash_valid(dataset):
        raise RuntimeError("frozen atlas dataset hash mismatch")
    started = time.perf_counter()
    model, _, metadata = _load_model(with_lens=False)
    rows = _behavior_rows(
        model,
        _select_rows(dataset, config, phase),
        int(config["behavior"]["batch_size"]),
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    cache.commit()
    result = {
        "metadata": {
            "schema_version": 1,
            "run_id": uuid.uuid4().hex,
            "created_at": datetime.now(UTC).isoformat(),
            "git_commit": git_commit,
            "phase": phase,
            "dataset_sha256": dataset["content_sha256"],
            "config_sha256": _canonical_sha256(config),
            "elapsed_seconds": elapsed,
            **metadata,
        },
        "summary": _behavior_summary(rows, config),
        "rows": rows,
    }
    return json.dumps(result, allow_nan=False, sort_keys=True)


def _deep_condition_ids(
    rows: list[dict[str, Any]], config: dict[str, Any], phase: str
) -> set[str]:
    if phase != "open":
        return set()
    count = int(config["deep_atlas"]["rows_per_game_per_open_split"])
    selected: set[str] = set()
    for game in config["games"]:
        for split in config["execution"]["open_splits"]:
            ids = sorted(
                row["condition_id"]
                for row in rows
                if row["game"] == game and row["split"] == split
            )
            selected.update(ids[:count])
    return selected


def _top_readout(model: Any, logits: Any, k: int) -> tuple[list[int], list[str], list[float]]:
    top = logits.topk(k)
    ids = [int(value) for value in top.indices.detach().cpu()]
    return (
        ids,
        [model.tokenizer.decode([token]) for token in ids],
        [float(v) for v in top.values.detach().cpu()],
    )


def _fit_probe_metrics(
    metadata: list[dict[str, Any]],
    residual_features: dict[int, Any],
    jspace_features: dict[int, list[dict[str, float]]],
    output_features: Any,
    layers: list[int],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    import numpy as np
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        mean_absolute_error,
        r2_score,
    )
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    output: list[dict[str, Any]] = []
    train_split = config["probe"]["training_split"]
    test_split = config["probe"]["evaluation_split"]

    def fit_one(
        game: str,
        variable: str,
        kind: str,
        representation: str,
        layer: int,
        features: Any,
    ) -> None:
        indices_train = [
            i
            for i, row in enumerate(metadata)
            if row["game"] == game and row["split"] == train_split
        ]
        indices_test = [
            i
            for i, row in enumerate(metadata)
            if row["game"] == game and row["split"] == test_split
        ]
        y_train = [metadata[i][variable] for i in indices_train]
        y_test = [metadata[i][variable] for i in indices_test]
        base = {
            "game": game,
            "variable": variable,
            "kind": kind,
            "representation": representation,
            "layer": layer,
            "n_train": len(indices_train),
            "n_test": len(indices_test),
        }
        if not indices_train or not indices_test:
            output.append({**base, "status": "missing_split"})
            return
        if kind == "categorical" and (len(set(y_train)) < 2 or not set(y_test) <= set(y_train)):
            output.append({**base, "status": "insufficient_classes"})
            return
        if kind == "scalar" and len(set(map(float, y_train))) < 2:
            output.append({**base, "status": "constant_target"})
            return
        if representation == "jspace":
            vectorizer = DictVectorizer(sparse=True)
            x_train = vectorizer.fit_transform([features[i] for i in indices_train])
            x_test = vectorizer.transform([features[i] for i in indices_test])
        else:
            x_train = np.asarray([features[i] for i in indices_train], dtype=np.float32)
            x_test = np.asarray([features[i] for i in indices_test], dtype=np.float32)
        if kind == "categorical":
            if representation == "residual":
                estimator = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        C=float(config["probe"]["regularization_c"]),
                        max_iter=int(config["probe"]["maximum_iterations"]),
                    ),
                )
            else:
                estimator = LogisticRegression(
                    C=float(config["probe"]["regularization_c"]),
                    max_iter=int(config["probe"]["maximum_iterations"]),
                )
            estimator.fit(x_train, y_train)
            predicted = estimator.predict(x_test)
            output.append(
                {
                    **base,
                    "status": "ok",
                    "accuracy": float(accuracy_score(y_test, predicted)),
                    "balanced_accuracy": float(balanced_accuracy_score(y_test, predicted)),
                }
            )
        else:
            if representation == "residual":
                estimator = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
            else:
                estimator = Ridge(alpha=10.0, solver="lsqr")
            estimator.fit(x_train, np.asarray(y_train, dtype=float))
            predicted = estimator.predict(x_test)
            true = np.asarray(y_test, dtype=float)
            correlation = (
                float(np.corrcoef(true, predicted)[0, 1]) if np.std(predicted) > 0 else 0.0
            )
            output.append(
                {
                    **base,
                    "status": "ok",
                    "r2": float(r2_score(true, predicted)),
                    "mae": float(mean_absolute_error(true, predicted)),
                    "correlation": correlation,
                }
            )

    categorical = list(config["probe"]["categorical_variables"])
    scalar = list(config["probe"]["scalar_variables"])
    for game in config["games"]:
        for variable in categorical + scalar:
            kind = "categorical" if variable in categorical else "scalar"
            fit_one(game, variable, kind, "output", -1, output_features)
            for layer in layers:
                fit_one(game, variable, kind, "residual", layer, residual_features[layer])
                fit_one(game, variable, kind, "jspace", layer, jspace_features[layer])
    return output


def _mechanistic_rows(
    model: Any,
    lens: Any,
    selected: list[dict[str, Any]],
    config: dict[str, Any],
    phase: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import numpy as np
    import torch
    from jlens import ActivationRecorder

    layers = [int(layer) for layer in lens.source_layers]
    deep_ids = _deep_condition_ids(selected, config, phase)
    top_k = int(config["top_k"])
    deep_k = int(config["deep_atlas"]["top_k"])
    deep_positions = int(config["deep_atlas"]["positions"])
    residual_features: dict[int, list[Any]] = {layer: [] for layer in layers}
    jspace_features: dict[int, list[dict[str, float]]] = {layer: [] for layer in layers}
    output_features: list[list[float]] = []
    feature_metadata: list[dict[str, Any]] = []
    output: list[dict[str, Any]] = []
    ordered = sorted(selected, key=lambda row: (row["sequence_length"], row["condition_id"]))
    batch_size = int(config["behavior"]["batch_size"])
    for start in range(0, len(ordered), batch_size):
        part = ordered[start : start + batch_size]
        input_ids, attention = _left_padded_batch(part)
        with torch.inference_mode(), ActivationRecorder(model.layers, at=layers) as recorder:
            model_output = model._hf_model(
                input_ids, attention_mask=attention, logits_to_keep=1
            )
        for index, row in enumerate(part):
            final_position = int(input_ids.shape[1]) - 1
            candidate_ids = [int(token) for token in row["candidate_token_ids"]]
            output_logits = model_output.logits[index, -1].float()
            legal_log_probs = output_logits[candidate_ids].log_softmax(-1)
            output_features.append([float(value.cpu()) for value in legal_log_probs])
            layer_rows: list[dict[str, Any]] = []
            for layer in layers:
                residual = recorder.activations[layer][index, final_position].detach().float()
                jacobian = lens.jacobians[layer].to(residual.device)
                transported = residual @ jacobian.T
                jlens_full = model.unembed(transported).float()
                logit_lens_full = model.unembed(residual).float()
                ids, texts, scores = _top_readout(model, jlens_full, top_k)
                rank_features = {
                    f"t{token}": 1.0 / (rank + 1) for rank, token in enumerate(ids)
                }
                legal_jlens = [float(jlens_full[token].cpu()) for token in candidate_ids]
                legal_logit_lens = [
                    float(logit_lens_full[token].cpu()) for token in candidate_ids
                ]
                layer_row: dict[str, Any] = {
                    "layer": layer,
                    "top_token_ids": ids,
                    "top_token_texts": texts,
                    "top_scores": scores,
                    "legal_action_jlens_scores": legal_jlens,
                    "legal_action_logit_lens_scores": legal_logit_lens,
                    "logit_lens_legal_choice": row["candidates"][
                        int(logit_lens_full[candidate_ids].argmax().cpu())
                    ],
                }
                if row["condition_id"] in deep_ids:
                    first = max(
                        final_position - deep_positions + 1,
                        final_position - int(row["sequence_length"]) + 1,
                    )
                    deep_residual = (
                        recorder.activations[layer][index, first : final_position + 1]
                        .detach()
                        .float()
                    )
                    deep_logits = model.unembed(deep_residual @ jacobian.T).float()
                    deep_top = deep_logits.topk(deep_k, dim=-1)
                    layer_row["deep_positions"] = [
                        {
                            "position_offset": position - deep_logits.shape[0] + 1,
                            "token_id": int(input_ids[index, first + position].cpu()),
                            "token_text": model.tokenizer.decode(
                                [int(input_ids[index, first + position].cpu())]
                            ),
                            "top_token_ids": [
                                int(value) for value in deep_top.indices[position].cpu()
                            ],
                            "top_token_texts": [
                                model.tokenizer.decode([int(value)])
                                for value in deep_top.indices[position].cpu()
                            ],
                            "top_scores": [
                                float(value) for value in deep_top.values[position].cpu()
                            ],
                        }
                        for position in range(deep_logits.shape[0])
                    ]
                layer_rows.append(layer_row)
                residual_features[layer].append(residual.cpu().numpy())
                jspace_features[layer].append(rank_features)
                del jacobian, transported, jlens_full, logit_lens_full
            feature_metadata.append(
                {
                    "condition_id": row["condition_id"],
                    "game": row["game"],
                    "split": row["split"],
                    "private_state": row["private_state"],
                    "belief": row["belief"],
                    "objective": row["objective"],
                    "value_margin": row["value_margin"],
                    "strategy": row["strategy"],
                    "action": row["expected_action"],
                }
            )
            output.append(
                {
                    "condition_id": row["condition_id"],
                    "matched_group_id": row["matched_group_id"],
                    "game": row["game"],
                    "split": row["split"],
                    "template_id": row["template_id"],
                    "task_text": row["task_text"],
                    "prompt_sha256": row["prompt_sha256"],
                    "private_state": row["private_state"],
                    "belief": row["belief"],
                    "objective": row["objective"],
                    "value_margin": row["value_margin"],
                    "strategy": row["strategy"],
                    "expected_action": row["expected_action"],
                    "output_top1_token_id": int(output_logits.argmax().cpu()),
                    "output_top1_text": model.tokenizer.decode(
                        [int(output_logits.argmax().cpu())]
                    ),
                    "output_legal_action_log_probs": [
                        float(value.cpu()) for value in legal_log_probs
                    ],
                    "deep_atlas": row["condition_id"] in deep_ids,
                    "layer_readouts": layer_rows,
                }
            )
        del input_ids, attention, model_output
    arrays = {
        layer: np.stack(values).astype("float32") for layer, values in residual_features.items()
    }
    metrics = _fit_probe_metrics(
        feature_metadata,
        arrays,
        jspace_features,
        np.asarray(output_features, dtype=np.float32),
        layers,
        config,
    )
    return sorted(output, key=lambda row: row["condition_id"]), metrics


@app.function(
    image=image,
    volumes={"/cache": cache},
    cpu=8.0,
    memory=65536,
    gpu="A100-80GB",
    timeout=7200,
    max_containers=1,
    retries=0,
)
def mechanistic_remote(
    phase: str,
    dataset: dict[str, Any],
    config: dict[str, Any],
    behavior_summary: dict[str, Any],
    git_commit: str,
) -> str:
    import torch

    if not _content_hash_valid(dataset):
        raise RuntimeError("frozen atlas dataset hash mismatch")
    if not behavior_summary.get("gate_pass"):
        raise RuntimeError("behavioral gate failed; mechanistic output remains unopened")
    started = time.perf_counter()
    model, lens, metadata = _load_model(with_lens=True)
    selected = _select_rows(dataset, config, phase)
    rows, probe_metrics = _mechanistic_rows(model, lens, selected, config, phase)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    cache.commit()
    result = {
        "metadata": {
            "schema_version": 1,
            "run_id": uuid.uuid4().hex,
            "created_at": datetime.now(UTC).isoformat(),
            "git_commit": git_commit,
            "phase": phase,
            "dataset_sha256": dataset["content_sha256"],
            "config_sha256": _canonical_sha256(config),
            "primary_position": config["primary_position"],
            "top_k": config["top_k"],
            "deep_atlas": config["deep_atlas"],
            "elapsed_seconds": elapsed,
            **metadata,
        },
        "probe_metrics": probe_metrics,
        "rows": rows,
    }
    return json.dumps(result, allow_nan=False, sort_keys=True)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _tracked_tree_clean() -> bool:
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], text=True
    )
    return not status.strip()


def _write_new(path: Path, text: str) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite prospective artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@app.local_entrypoint()
def freeze_dataset() -> None:
    config = _load_json(Path("configs/v2/workspace_atlas/experiment.json"))
    payload = tokenize_dataset_remote.remote(config)
    output = Path("configs/v2/workspace_atlas/dataset.json")
    _write_new(output, payload + "\n")
    parsed = json.loads(payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "content_sha256": parsed["content_sha256"],
                "n_rows": len(parsed["rows"]),
                "n_matched_groups": len({row["matched_group_id"] for row in parsed["rows"]}),
            },
            indent=2,
        )
    )


@app.local_entrypoint()
def behavior(phase: str = "open") -> None:
    if phase not in {"open", "locked"}:
        raise ValueError("phase must be open or locked")
    config = _load_json(Path("configs/v2/workspace_atlas/experiment.json"))
    dataset = _load_json(Path("configs/v2/workspace_atlas/dataset.json"))
    output = Path(f"results/v2_workspace_atlas/raw/behavior_{phase}.json")
    estimate = estimate_cost("A100-80GB", 900, memory_gib=32)
    ledger = Path("results/v2_workspace_atlas/cost_ledger.jsonl")
    admit_run(
        ledger, estimate, study_limit_usd=float(config["execution"]["study_cost_limit_usd"])
    )
    payload = behavior_remote.remote(phase, dataset, config, _git_head())
    parsed = json.loads(payload)
    _write_new(output, json.dumps(parsed, indent=2, sort_keys=True) + "\n")
    measured = estimate_cost(
        "A100-80GB", float(parsed["metadata"]["elapsed_seconds"]), memory_gib=32
    )
    append_ledger(
        ledger, measured, run_id=parsed["metadata"]["run_id"], stage=f"atlas_behavior_{phase}"
    )
    print(json.dumps(parsed["summary"], indent=2, sort_keys=True))


@app.local_entrypoint()
def mechanistic(phase: str = "open") -> None:
    if phase not in {"open", "locked"}:
        raise ValueError("phase must be open or locked")
    if not _tracked_tree_clean():
        raise RuntimeError("mechanistic execution requires a clean tracked worktree")
    config = _load_json(Path("configs/v2/workspace_atlas/experiment.json"))
    dataset = _load_json(Path("configs/v2/workspace_atlas/dataset.json"))
    behavior_result = _load_json(Path(f"results/v2_workspace_atlas/raw/behavior_{phase}.json"))
    output = Path(f"results/v2_workspace_atlas/raw/mechanistic_{phase}.json")
    estimate = estimate_cost("A100-80GB", 7200, cpu_cores=8, memory_gib=64)
    ledger = Path("results/v2_workspace_atlas/cost_ledger.jsonl")
    admit_run(
        ledger, estimate, study_limit_usd=float(config["execution"]["study_cost_limit_usd"])
    )
    payload = mechanistic_remote.remote(
        phase, dataset, config, behavior_result["summary"], _git_head()
    )
    parsed = json.loads(payload)
    _write_new(output, json.dumps(parsed, indent=2, sort_keys=True) + "\n")
    measured = estimate_cost(
        "A100-80GB", float(parsed["metadata"]["elapsed_seconds"]), cpu_cores=8, memory_gib=64
    )
    append_ledger(
        ledger,
        measured,
        run_id=parsed["metadata"]["run_id"],
        stage=f"atlas_mechanistic_{phase}",
    )
    print(
        json.dumps(
            {
                "phase": phase,
                "rows": len(parsed["rows"]),
                "probe_metrics": len(parsed["probe_metrics"]),
                "elapsed_seconds": parsed["metadata"]["elapsed_seconds"],
            },
            indent=2,
        )
    )
