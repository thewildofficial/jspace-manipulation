"""Prospective V5-RBG-5 behavior, discovery, locked, and optional J-space runs.

Commands are deliberately separated so the locked endpoint cannot open before a
valid behavioral gate and a content-addressed discovery freeze::

    modal run modal_v5_mechanistic_decomposition.py::preflight
    modal run modal_v5_mechanistic_decomposition.py::behavior
    modal run modal_v5_mechanistic_decomposition.py::discovery
    modal run modal_v5_mechanistic_decomposition.py::locked
    modal run modal_v5_mechanistic_decomposition.py::jspace
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

from jspace_policy.budget import append_ledger, estimate_cost, ledger_total

CONFIG_PATH = Path("configs/v5/mechanistic_decomposition/experiment.json")
DATASET_PATH = Path("configs/v5/mechanistic_decomposition/dataset.json")
MANIFEST_PATH = Path("configs/v5/mechanistic_decomposition/dataset_manifest.json")
RESULT_ROOT = Path("results/v5_mechanistic_decomposition")
MODEL_ID = "Qwen/Qwen3.6-27B"
MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
DATASET_SHA256 = "238ac7d6a36c49851ef2ebabce201aca3ad3c33c2d79251ab67bf2ee16090c8e"
SYSTEM_PROMPT = (
    "Treat the described game and prior episodes as exact formal information. "
    "Follow the requested forced-choice format exactly."
)

app = modal.App("jspace-v5-mechanistic-decomposition")
cache = modal.Volume.from_name("jspace-hf-cache", create_if_missing=True)
artifacts = modal.Volume.from_name("jspace-v5-rbg5-artifacts", create_if_missing=True)
base_image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .uv_pip_install(
        "numpy>=2.0",
        "scikit-learn>=1.6",
        "torch>=2.8",
        "transformers>=5.5",
        "huggingface_hub>=0.34",
    )
    .env({"HF_HOME": "/cache/huggingface", "TOKENIZERS_PARALLELISM": "false"})
)
core_image = base_image.add_local_dir(
    "src/jspace_policy", remote_path="/root/jspace_policy"
)
JLENS_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
lens_image = base_image.uv_pip_install(
    f"git+https://github.com/anthropics/jacobian-lens.git@{JLENS_COMMIT}"
).add_local_dir(
    "src/jspace_policy", remote_path="/root/jspace_policy"
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _study_id(config: dict[str, Any]) -> str:
    return str(config.get("study_id", "V5-RBG-5"))


def _artifact_prefix(config: dict[str, Any]) -> str:
    return str(config.get("execution", {}).get("artifact_prefix", "rbg5"))


def _result_root(config: dict[str, Any]) -> Path:
    return Path(
        config.get("execution", {}).get(
            "result_root", "results/v5_mechanistic_decomposition"
        )
    )


def _expected_dataset_sha256(config: dict[str, Any]) -> str:
    expected = config.get("dataset", {}).get("expected_content_sha256")
    return str(expected or DATASET_SHA256)


def _write_new(path: Path, text: str) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite prospective artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _tracked_tree_clean() -> bool:
    result = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], text=True
    )
    return not result.strip()


def _validate(config: dict[str, Any], dataset: dict[str, Any]) -> None:
    from jspace_policy.mechanistic_decomposition_games import verify_dataset_payload

    expected_status = (
        "prospectively_frozen_after_rbg4_before_dataset_materialization_or_model_execution"
    )
    if config["status"] not in {
        expected_status,
        "prospectively_frozen_before_dataset_materialization_or_model_execution",
    }:
        raise RuntimeError("protocol is not prospectively frozen")
    if config["model"]["id"] != MODEL_ID or config["model"]["revision"] != MODEL_REVISION:
        raise RuntimeError("pinned model changed")
    verify_dataset_payload(dataset, config)
    expected_dataset = _expected_dataset_sha256(config)
    if dataset["content_sha256"] != expected_dataset:
        raise RuntimeError("materialized dataset does not match prospective manifest")
    if "expected_content_sha256" not in config.get("dataset", {}) and MANIFEST_PATH.exists():
        manifest = _load_json(MANIFEST_PATH)
        if manifest["expected_content_sha256"] != expected_dataset:
            raise RuntimeError("local prospective manifest does not match pinned dataset hash")


def _content_hash_valid(payload: dict[str, Any]) -> bool:
    claimed = payload.get("content_sha256")
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    return claimed == _canonical_sha256(body)


def _continuation_id(tokenizer: Any, rendered: str, answer: str) -> int:
    prefix = tokenizer.encode(rendered, add_special_tokens=False)
    full = tokenizer.encode(rendered + answer, add_special_tokens=False)
    if full[: len(prefix)] == prefix and len(full) == len(prefix) + 1:
        return int(full[-1])
    raise ValueError(f"{answer!r} is not one token after the frozen Answer: prefix")


def _messages(text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


def _render_query(
    tokenizer: Any,
    query_id: str,
    text: str,
    candidates: list[str],
    anchor_char_ends: dict[str, int],
) -> dict[str, Any]:
    rendered = tokenizer.apply_chat_template(
        _messages(text), tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    token_ids = list(map(int, encoded["input_ids"]))
    offsets = [tuple(map(int, pair)) for pair in encoded["offset_mapping"]]
    user_start = rendered.find(text)
    if user_start < 0:
        raise RuntimeError("rendered chat template did not preserve user text")
    anchor_positions = {}
    for name, char_end in anchor_char_ends.items():
        absolute_end = user_start + int(char_end)
        # Section boundaries are represented by the first token of the common
        # separator/next section.  Taking the last token of a variable-length
        # prose/table section can select punctuation that differs by format
        # (and therefore cannot be used for same-base interchange).  History
        # episode anchors remain end-of-episode tokens because their response
        # sentence is the shared semantic suffix.
        if name in {"mapping_end", "actions_end", "payoff_end"}:
            candidates_after = [
                index
                for index, (start, end) in enumerate(offsets)
                if end > start >= absolute_end
            ]
            if candidates_after:
                anchor_positions[name] = candidates_after[0]
                continue
        candidates_at_or_before = [
            index for index, (_, end) in enumerate(offsets) if end and end <= absolute_end
        ]
        if not candidates_at_or_before:
            raise RuntimeError(f"could not resolve semantic anchor {name}")
        anchor_positions[name] = candidates_at_or_before[-1]
    return {
        "query_id": query_id,
        "rendered": rendered,
        "prompt_token_ids": token_ids,
        "sequence_length": len(token_ids),
        "candidate_labels": candidates,
        "candidate_token_ids": [
            _continuation_id(tokenizer, rendered, candidate) for candidate in candidates
        ],
        "anchor_positions": anchor_positions,
    }


def _action_query(tokenizer: Any, row: dict[str, Any]) -> dict[str, Any]:
    return _render_query(
        tokenizer,
        row["condition_id"],
        row["prompt"],
        ["A", "B"],
        row["anchor_char_ends"],
    )


def _report_query(
    tokenizer: Any, row: dict[str, Any], action: str
) -> tuple[dict[str, Any], str]:
    from jspace_policy.mechanistic_decomposition_games import report_question

    question, expected = report_question(row, action)
    scenario = row["prompt"][: row["anchor_char_ends"]["payoff_end"]]
    text = f"{scenario}\n{question}"
    anchors = {
        key: value
        for key, value in row["anchor_char_ends"].items()
        if value <= row["anchor_char_ends"]["payoff_end"]
    }
    anchors["answer"] = len(text)
    query = _render_query(
        tokenizer, f"{row['condition_id']}:{action}", text, ["X", "Y"], anchors
    )
    return query, expected


def _left_padded(
    queries: list[dict[str, Any]], pad_token_id: int
) -> tuple[Any, Any, list[int]]:
    import torch

    width = max(query["sequence_length"] for query in queries)
    input_ids = torch.full(
        (len(queries), width), pad_token_id, dtype=torch.long, device="cuda"
    )
    attention = torch.zeros_like(input_ids)
    pads = []
    for index, query in enumerate(queries):
        tokens = torch.tensor(query["prompt_token_ids"], dtype=torch.long, device="cuda")
        pad = width - len(tokens)
        pads.append(pad)
        input_ids[index, pad:] = tokens
        attention[index, pad:] = 1
    return input_ids, attention, pads


def _model_layers(model: Any) -> Any:
    candidates = [
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(getattr(model, "model", None), "language_model", None), "layers", None),
    ]
    for layers in candidates:
        if layers is not None:
            return layers
    raise RuntimeError("could not locate transformer layers")


def _capture_row_allowed(row: dict[str, Any], config: dict[str, Any]) -> bool:
    cells = config.get("dataset", {}).get("capture_cells")
    if not cells:
        return True
    key = [row["incentive"], row["surface_kind"], row["history"], row["mapping_format"]]
    return key in cells


def _decode_residual_slice(values: Any, *, device: str = "cpu") -> Any:
    """Decode either legacy float16 or exact BF16 uint16 storage."""
    import torch

    if getattr(values, "dtype", None) is not None and str(values.dtype) == "uint16":
        tensor = torch.from_numpy(values)
        return tensor.view(torch.bfloat16).to(device=device)
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def _load_model() -> tuple[Any, Any, dict[str, Any]]:
    import torch
    import transformers
    from huggingface_hub import model_info

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).cuda()
    model.eval()
    layers = _model_layers(model)
    if len(layers) != 64:
        raise RuntimeError(f"pinned model layer count changed: {len(layers)}")
    width = int(model.config.hidden_size)
    if width != 5120:
        raise RuntimeError(f"pinned residual width changed: {width}")
    resolved = model_info(MODEL_ID, revision=MODEL_REVISION).sha
    metadata = {
        "model_id": MODEL_ID,
        "model_revision_requested": MODEL_REVISION,
        "model_revision_resolved": resolved,
        "tokenizer_revision": resolved,
        "dtype": "bfloat16",
        "n_layers": len(layers),
        "d_model": width,
        "gpu_actual": torch.cuda.get_device_name(0),
        "torch_version": str(torch.__version__),
        "transformers_version": transformers.__version__,
    }
    return model, tokenizer, metadata


def _query_batches(
    model: Any,
    tokenizer: Any,
    queries: list[dict[str, Any]],
    batch_size: int,
) -> dict[str, dict[str, Any]]:
    import torch

    output = {}
    ordered = sorted(queries, key=lambda row: (row["sequence_length"], row["query_id"]))
    for start in range(0, len(ordered), batch_size):
        part = ordered[start : start + batch_size]
        input_ids, attention, _ = _left_padded(part, tokenizer.pad_token_id)
        with torch.inference_mode():
            logits = model(
                input_ids=input_ids, attention_mask=attention, logits_to_keep=1
            ).logits[:, -1].float()
        for index, query in enumerate(part):
            final = logits[index]
            candidate_ids = list(map(int, query["candidate_token_ids"]))
            legal = final[candidate_ids]
            choice = int(legal.argmax().cpu())
            top1 = int(final.argmax().cpu())
            output[query["query_id"]] = {
                "legal_choice": query["candidate_labels"][choice],
                "legal_logits": dict(
                    zip(
                        query["candidate_labels"],
                        map(float, legal.detach().cpu()),
                        strict=True,
                    )
                ),
                "top1_token_id": top1,
                "top1_text": tokenizer.decode([top1]),
                "formatting_compliant": top1 in candidate_ids,
            }
        del input_ids, attention, logits
    return output


@app.function(image=core_image, cpu=2, memory=8192, volumes={"/cache": cache}, timeout=900)
def preflight_remote(dataset: dict[str, Any], config: dict[str, Any]) -> str:
    import transformers

    from jspace_policy.mechanistic_decomposition_games import matched_row

    _validate(config, dataset)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    checked = []
    for row in dataset["rows"]:
        action = _action_query(tokenizer, row)
        checked.append((action["sequence_length"], len(action["anchor_positions"])))
        for option in ("A", "B"):
            report, _ = _report_query(tokenizer, row, option)
            checked.append((report["sequence_length"], len(report["anchor_positions"])))
    paired_anchor_pairs_checked = 0
    if config.get("capture", {}).get("require_paired_anchor_token_ids", False):
        action_queries = {
            row["condition_id"]: _action_query(tokenizer, row) for row in dataset["rows"]
        }
        recipients = [
            row for row in dataset["rows"]
            if row["incentive"] == "opposed"
            and row["surface_kind"] == "assertion"
            and row["history"] == "redundant"
            and row["mapping_format"] == "prose"
        ]
        for recipient in recipients:
            for family in ("table", "opaque"):
                donor = matched_row(dataset["rows"], recipient, family)
                left = action_queries[recipient["condition_id"]]
                right = action_queries[donor["condition_id"]]
                for anchor in config["patch"]["candidate_anchors"]:
                    left_id = left["prompt_token_ids"][left["anchor_positions"][anchor]]
                    right_id = right["prompt_token_ids"][right["anchor_positions"][anchor]]
                    if left_id != right_id:
                        raise RuntimeError(
                            "paired anchor token mismatch for "
                            f"{anchor}: {recipient['condition_id']}"
                        )
                paired_anchor_pairs_checked += 1
    cache.commit()
    return json.dumps(
        {
            "schema_version": 1,
            "study_id": _study_id(config),
            "created_at": datetime.now(UTC).isoformat(),
            "dataset_sha256": dataset["content_sha256"],
            "queries_checked": len(checked),
            "minimum_tokens": min(item[0] for item in checked),
            "maximum_tokens": max(item[0] for item in checked),
            "minimum_anchors": min(item[1] for item in checked),
            "paired_anchor_pairs_checked": paired_anchor_pairs_checked,
            "status": "all_token_continuation_and_semantic_anchor_checks_passed",
        },
        sort_keys=True,
    )


@app.function(
    image=core_image,
    gpu="A100-80GB",
    cpu=8,
    memory=32768,
    volumes={"/cache": cache},
    timeout=2400,
    retries=0,
)
def behavior_remote(dataset: dict[str, Any], config: dict[str, Any], git_commit: str) -> str:
    import torch

    from jspace_policy.mechanistic_decomposition import analyze_behavior

    _validate(config, dataset)
    torch.manual_seed(int(config["execution"]["seed"]))
    started = time.perf_counter()
    model, tokenizer, model_metadata = _load_model()
    action_queries = [_action_query(tokenizer, row) for row in dataset["rows"]]
    actions = _query_batches(
        model, tokenizer, action_queries, int(config["execution"]["behavior_batch_size"])
    )
    report_queries = []
    expected_reports = {}
    for row in dataset["rows"]:
        for option in ("A", "B"):
            query, expected = _report_query(tokenizer, row, option)
            report_queries.append(query)
            expected_reports[query["query_id"]] = expected
    reports = _query_batches(
        model, tokenizer, report_queries, int(config["execution"]["behavior_batch_size"])
    )
    rows = []
    for row in dataset["rows"]:
        condition_id = row["condition_id"]
        selected = actions[condition_id]["legal_choice"]
        option_reports = {}
        for option in ("A", "B"):
            query_id = f"{condition_id}:{option}"
            result = reports[query_id]
            expected = expected_reports[query_id]
            option_reports[option] = {
                "choice": result["legal_choice"],
                "expected": expected,
                "correct": result["legal_choice"] == expected,
                "result": result,
            }
        rows.append(
            {
                **row,
                "selected_action": selected,
                "chosen_action_index": ["A", "B"].index(selected),
                "action_correct": selected == row["expected_action"],
                "action_result": actions[condition_id],
                "option_reports": option_reports,
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
            "git_commit": git_commit,
            "dataset_sha256": dataset["content_sha256"],
            "config_sha256": _canonical_sha256(config),
            **model_metadata,
        },
        "rows": sorted(rows, key=lambda row: row["condition_id"]),
    }
    payload["analysis"] = analyze_behavior(payload, config)
    cache.commit()
    return json.dumps(payload, allow_nan=False, sort_keys=True)


def _capture_split(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    behavior_rows: dict[str, dict[str, Any]],
    config: dict[str, Any],
    output_path: Path,
) -> tuple[list[dict[str, Any]], list[str], list[int]]:
    import numpy as np
    import torch

    layers = _model_layers(model)
    anchor_names = list(config["capture"]["anchors"])
    width = int(model.config.hidden_size)
    ordered = sorted(rows, key=lambda row: row["condition_id"])
    storage_dtype = str(config["capture"].get("storage_dtype", "float16"))
    if storage_dtype == "bfloat16_bits_uint16":
        numpy_dtype = np.uint16
    elif storage_dtype == "float16":
        numpy_dtype = np.float16
    else:
        raise ValueError(f"unsupported activation storage dtype: {storage_dtype}")
    features = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=numpy_dtype,
        shape=(len(ordered), len(layers), len(anchor_names), width),
    )
    masks = np.zeros((len(ordered), len(anchor_names)), dtype=bool)
    metadata = []
    queries = [_action_query(tokenizer, row) for row in ordered]
    batch_size = int(config["execution"]["capture_batch_size"])
    for start in range(0, len(ordered), batch_size):
        part_rows = ordered[start : start + batch_size]
        part_queries = queries[start : start + batch_size]
        input_ids, attention, pads = _left_padded(part_queries, tokenizer.pad_token_id)
        positions = []
        for index, query in enumerate(part_queries):
            row_positions = []
            for anchor_index, anchor in enumerate(anchor_names):
                present = anchor in query["anchor_positions"]
                masks[start + index, anchor_index] = present
                relative = query["anchor_positions"].get(
                    anchor, query["anchor_positions"]["history_end"]
                )
                row_positions.append(pads[index] + int(relative))
            positions.append(row_positions)
        position_tensor = torch.tensor(positions, dtype=torch.long, device="cuda")
        batch_indices = torch.arange(len(part_rows), device="cuda")[:, None]
        handles = []

        def make_hook(
            layer_index: int,
            *,
            selected_indices: Any = batch_indices,
            selected_positions: Any = position_tensor,
            start_index: int = start,
            part_length: int = len(part_rows),
        ) -> Any:
            def hook(_module: Any, _inputs: Any, output: Any) -> Any:
                hidden = output[0] if isinstance(output, tuple) else output
                selected = hidden[selected_indices, selected_positions].detach()
                if storage_dtype == "bfloat16_bits_uint16":
                    selected = selected.to(device="cpu", dtype=torch.bfloat16)
                    selected = selected.view(torch.uint16)
                else:
                    selected = selected.to(device="cpu", dtype=torch.float16)
                features[
                    start_index : start_index + part_length, layer_index, :, :
                ] = selected.numpy()
                return output

            return hook

        for layer_index, layer in enumerate(layers):
            handles.append(layer.register_forward_hook(make_hook(layer_index)))
        try:
            with torch.inference_mode():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention,
                    logits_to_keep=1,
                )
        finally:
            for handle in handles:
                handle.remove()
        for index, (row, query) in enumerate(zip(part_rows, part_queries, strict=True)):
            final = outputs.logits[index, -1].float()
            legal_ids = list(map(int, query["candidate_token_ids"]))
            legal = final[legal_ids]
            observed = behavior_rows[row["condition_id"]]
            metadata.append(
                {
                    "condition_id": row["condition_id"],
                    "base_game_id": row["base_game_id"],
                    "split": row["split"],
                    "frame": row["frame"],
                    "incentive": row["incentive"],
                    "surface_kind": row["surface_kind"],
                    "history": row["history"],
                    "mapping_format": row["mapping_format"],
                    "option_a_response_index": row["option_a_response_index"],
                    "target_index": row["target_index"],
                    "correct_action_index": row["correct_action_index"],
                    "chosen_action_index": observed["chosen_action_index"],
                    "expected_action": row["expected_action"],
                    "selected_action": observed["selected_action"],
                    "action_correct": observed["action_correct"],
                    "legal_logits": dict(zip(["A", "B"], map(float, legal.cpu()), strict=True)),
                }
            )
        del input_ids, attention, outputs, position_tensor
    features.flush()
    mask_path = output_path.with_name("anchor_mask.npy")
    np.save(mask_path, masks, allow_pickle=False)
    return metadata, anchor_names, list(range(len(layers)))


def _decode_numpy_features(values: Any) -> Any:
    """Convert exact BF16 uint16 storage (or legacy float16) to float32."""
    import numpy as np

    array = np.asarray(values)
    if array.dtype == np.uint16:
        return (array.astype(np.uint32) << 16).view(np.float32)
    return array.astype(np.float32, copy=False)


def _probe_layer_indices(anchor: str, layers: list[int], config: dict[str, Any]) -> list[int]:
    probe = config.get("probe", {})
    if anchor in probe.get("history_anchors", []):
        requested = probe.get("history_probe_layers", layers)
    else:
        requested = probe.get("core_probe_layers", layers)
    allowed = set(layers)
    return [int(layer) for layer in requested if int(layer) in allowed]


def _fit_probe_artifact(
    residual_path: Path,
    mask_path: Path,
    metadata: list[dict[str, Any]],
    anchor_names: list[str],
    layers: list[int],
    config: dict[str, Any],
    remote_root: Path,
    dataset_sha256: str,
) -> dict[str, Any]:
    import pickle

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    features = np.load(residual_path, mmap_mode="r")
    masks = np.load(mask_path)
    targets = list(config["probe"]["targets"])
    fixed_c = config["probe"].get("fixed_regularization_c")
    c_grid = [float(fixed_c)] if fixed_c is not None else list(
        map(float, config["probe"]["regularization_c_grid"])
    )
    # RBG-5B fixes C prospectively and uses the high-dimensional dual L2
    # solver.  Keep the legacy RBG-5 path's liblinear solver unchanged.
    fixed_solver = str(config["probe"].get("solver", "liblinear"))
    folds = int(config["probe"]["group_folds"])
    rng = np.random.default_rng(int(config["probe"]["seed"]))
    models = []
    selection = []
    for anchor_index, anchor in enumerate(anchor_names):
        for layer in _probe_layer_indices(anchor, layers, config):
            valid = np.flatnonzero(masks[:, anchor_index])
            for target in targets:
                y_all = np.asarray([int(row[target]) for row in metadata])
                populations = config.get("probe", {}).get("target_training_population", {})
                configured_population = populations.get(target)
                model_kinds = (
                    [configured_population]
                    if configured_population
                    else ["all", "successful_formats"]
                )
                for model_kind in model_kinds:
                    if model_kind in {"successful_formats", "donor_formats"}:
                        valid_kind = np.asarray(
                            [
                                index
                                for index in valid
                                if metadata[index]["mapping_format"] == "table"
                                or metadata[index]["surface_kind"] == "opaque_token"
                            ],
                            dtype=int,
                        )
                    else:
                        valid_kind = valid
                    x = _decode_numpy_features(features[valid_kind, layer, anchor_index])
                    y = y_all[valid_kind]
                    groups = np.asarray(
                        [metadata[index]["base_game_id"] for index in valid_kind]
                    )
                    if len(set(y)) < 2 or len(set(groups)) < folds:
                        selection.append(
                            {
                                "layer": layer,
                                "anchor": anchor,
                                "target": target,
                                "model_kind": model_kind,
                                "status": "insufficient_classes_or_groups",
                            }
                        )
                        continue
                    # RBG-5B fixes C prospectively and has no hyperparameter
                    # search.  Fit each probe once on discovery data; locked
                    # evaluation is the preregistered out-of-sample readout.
                    # The old RBG-5 path below retains its frozen CV/grid logic.
                    if fixed_c is not None:
                        estimator = make_pipeline(
                            StandardScaler(),
                            LogisticRegression(
                                C=float(fixed_c),
                                solver=fixed_solver,
                                dual=fixed_solver == "liblinear",
                                class_weight="balanced",
                                max_iter=1000,
                                random_state=int(config["probe"]["seed"]),
                            ),
                        )
                        estimator.fit(x, y)
                        models.append(
                            {
                                "layer": layer,
                                "anchor": anchor,
                                "anchor_index": anchor_index,
                                "target": target,
                                "model_kind": model_kind,
                                "estimator": estimator,
                            }
                        )
                        selection.append(
                            {
                                "layer": layer,
                                "anchor": anchor,
                                "target": target,
                                "model_kind": model_kind,
                                "status": "ok",
                                "n": len(valid_kind),
                                "selected_c": float(fixed_c),
                                "solver": fixed_solver,
                                "discovery_fit_balanced_accuracy": float(
                                    balanced_accuracy_score(y, estimator.predict(x))
                                ),
                            }
                        )
                        continue
                    splitter = GroupKFold(n_splits=folds)
                    scores_by_c = {}
                    for regularization_c in c_grid:
                        fold_scores = []
                        for train, test in splitter.split(x, y, groups):
                            estimator = make_pipeline(
                                StandardScaler(),
                                LogisticRegression(
                                    C=regularization_c,
                                    solver="liblinear",
                                    dual=True,
                                    class_weight="balanced",
                                    max_iter=1000,
                                    random_state=int(config["probe"]["seed"]),
                                ),
                            )
                            estimator.fit(x[train], y[train])
                            fold_scores.append(
                                balanced_accuracy_score(y[test], estimator.predict(x[test]))
                            )
                        scores_by_c[regularization_c] = float(np.mean(fold_scores))
                    selected_c = sorted(c_grid, key=lambda c: (-scores_by_c[c], c))[0]
                    estimator = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(
                            C=selected_c,
                            solver="liblinear",
                            dual=True,
                            class_weight="balanced",
                            max_iter=1000,
                            random_state=int(config["probe"]["seed"]),
                        ),
                    )
                    estimator.fit(x, y)
                    shuffled = rng.permutation(y)
                    shuffled_scores = []
                    for train, test in splitter.split(x, shuffled, groups):
                        control = make_pipeline(
                            StandardScaler(),
                            LogisticRegression(
                                C=selected_c,
                                solver="liblinear",
                                dual=True,
                                class_weight="balanced",
                                max_iter=1000,
                                random_state=int(config["probe"]["seed"]),
                            ),
                        )
                        control.fit(x[train], shuffled[train])
                        shuffled_scores.append(
                            balanced_accuracy_score(
                                shuffled[test], control.predict(x[test])
                            )
                        )
                    models.append(
                        {
                            "layer": layer,
                            "anchor": anchor,
                            "anchor_index": anchor_index,
                            "target": target,
                            "model_kind": model_kind,
                            "estimator": estimator,
                        }
                    )
                    selection.append(
                        {
                            "layer": layer,
                            "anchor": anchor,
                            "target": target,
                            "model_kind": model_kind,
                            "status": "ok",
                            "n": len(valid_kind),
                            "selected_c": selected_c,
                            "cv_balanced_accuracy": scores_by_c[selected_c],
                            "shuffled_cv_balanced_accuracy": float(np.mean(shuffled_scores)),
                            "scores_by_c": {
                                str(key): value for key, value in scores_by_c.items()
                            },
                        }
                    )
    # Frozen, low-cost shuffled-label controls for four sentinel sites.
    sentinel_sites = config.get("probe", {}).get("label_shuffle_sentinel_sites", [])
    n_permutations = int(config.get("probe", {}).get("label_shuffle_permutations", 0))
    if sentinel_sites and n_permutations:
        populations = config.get("probe", {}).get("target_training_population", {})
        for layer, anchor in sentinel_sites:
            if anchor not in anchor_names or int(layer) not in layers:
                continue
            anchor_index = anchor_names.index(anchor)
            valid = np.flatnonzero(masks[:, anchor_index])
            for target in targets:
                population = populations.get(target)
                if population in {"successful_formats", "donor_formats"}:
                    valid_kind = np.asarray(
                        [
                            index for index in valid
                            if metadata[index]["mapping_format"] == "table"
                            or metadata[index]["surface_kind"] == "opaque_token"
                        ],
                        dtype=int,
                    )
                else:
                    valid_kind = valid
                x = _decode_numpy_features(features[valid_kind, int(layer), anchor_index])
                y = np.asarray([int(metadata[index][target]) for index in valid_kind])
                groups = np.asarray([metadata[index]["base_game_id"] for index in valid_kind])
                if len(set(y)) < 2 or len(set(groups)) < folds:
                    continue
                splitter = GroupKFold(n_splits=folds)
                for permutation in range(n_permutations):
                    shuffled = rng.permutation(y)
                    scores = []
                    for train, test in splitter.split(x, shuffled, groups):
                        control = make_pipeline(
                            StandardScaler(),
                            LogisticRegression(
                                C=c_grid[0],
                                solver=fixed_solver,
                                dual=fixed_solver == "liblinear",
                                class_weight="balanced",
                                max_iter=1000,
                                random_state=int(config["probe"]["seed"]) + permutation,
                            ),
                        )
                        control.fit(x[train], shuffled[train])
                        scores.append(
                            balanced_accuracy_score(
                                shuffled[test], control.predict(x[test])
                            )
                        )
                    selection.append({
                        "layer": int(layer),
                        "anchor": anchor,
                        "target": target,
                        "model_kind": population or "all",
                        "status": "shuffled_label_control",
                        "permutation": permutation,
                        "balanced_accuracy": float(np.mean(scores)),
                    })
    model_path = remote_root / "probe_models.pkl.gz"
    with gzip.open(model_path, "wb") as handle:
        pickle.dump(models, handle, protocol=pickle.HIGHEST_PROTOCOL)
    artifact = {
        "schema_version": 1,
        "study_id": _study_id(config),
        "status": "frozen_after_discovery_before_locked_capture",
        "dataset_sha256": dataset_sha256,
        "remote_model_path": str(model_path),
        "remote_model_sha256": _sha256_file(model_path),
        "selection": selection,
    }
    artifact["content_sha256"] = _canonical_sha256(artifact)
    return artifact


def _patched_query_batch(
    model: Any,
    tokenizer: Any,
    queries: list[dict[str, Any]],
    layer_index: int,
    anchor: str,
    donor_vectors: Any,
) -> list[dict[str, float]]:
    import torch

    input_ids, attention, pads = _left_padded(queries, tokenizer.pad_token_id)
    positions = torch.tensor(
        [
            pads[index] + query["anchor_positions"][anchor]
            for index, query in enumerate(queries)
        ],
        dtype=torch.long,
        device="cuda",
    )
    if (
        getattr(donor_vectors, "dtype", None) is not None
        and str(donor_vectors.dtype) == "uint16"
    ):
        vectors = torch.from_numpy(donor_vectors).view(torch.bfloat16).to(device="cuda")
    else:
        vectors = torch.as_tensor(donor_vectors, device="cuda", dtype=torch.bfloat16)
    batch_indices = torch.arange(len(queries), device="cuda")
    layer = _model_layers(model)[layer_index]

    def patch(
        _module: Any,
        _inputs: Any,
        output: Any,
        *,
        patch_indices: Any = batch_indices,
        patch_positions: Any = positions,
        patch_vectors: Any = vectors,
    ) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        modified = hidden.clone()
        modified[patch_indices, patch_positions] = patch_vectors
        if isinstance(output, tuple):
            return (modified, *output[1:])
        return modified

    handle = layer.register_forward_hook(patch)
    try:
        with torch.inference_mode():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention,
                logits_to_keep=1,
            ).logits[:, -1].float()
    finally:
        handle.remove()
    results = []
    for index, query in enumerate(queries):
        ids = list(map(int, query["candidate_token_ids"]))
        values = logits[index, ids]
        results.append(
            dict(zip(query["candidate_labels"], map(float, values.cpu()), strict=True))
        )
    del input_ids, attention, logits, vectors
    return results


def _run_discovery_patches(
    model: Any,
    tokenizer: Any,
    dataset_rows: list[dict[str, Any]],
    behavior_rows: dict[str, dict[str, Any]],
    residual_path: Path,
    metadata: list[dict[str, Any]],
    anchor_names: list[str],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    import numpy as np

    from jspace_policy.mechanistic_decomposition import correct_action_margin
    from jspace_policy.mechanistic_decomposition_games import matched_row

    residuals = np.load(residual_path, mmap_mode="r")
    feature_index = {row["condition_id"]: index for index, row in enumerate(metadata)}
    discovery_rows = [row for row in dataset_rows if row["split"] == "discovery"]
    recipients = [
        row
        for row in discovery_rows
        if row["incentive"] == "opposed"
        and row["surface_kind"] == "assertion"
        and row["history"] == "redundant"
        and row["mapping_format"] == "prose"
        and not behavior_rows[row["condition_id"]]["action_correct"]
        and all(
            report["correct"]
            for report in behavior_rows[row["condition_id"]]["option_reports"].values()
        )
    ]
    output = []
    batch_size = int(config["execution"]["patch_batch_size"])
    for donor_family in config["patch"]["donor_families"]:
        paired = []
        for recipient in recipients:
            donor = matched_row(discovery_rows, recipient, donor_family)
            if behavior_rows[donor["condition_id"]]["action_correct"]:
                paired.append((recipient, donor))
        for layer in map(int, config["patch"]["candidate_layers"]):
            for anchor in config["patch"]["candidate_anchors"]:
                anchor_index = anchor_names.index(anchor)
                for start in range(0, len(paired), batch_size):
                    part = paired[start : start + batch_size]
                    queries = [_action_query(tokenizer, recipient) for recipient, _ in part]
                    vectors = np.stack(
                        [
                            residuals[feature_index[donor["condition_id"]], layer, anchor_index]
                            for _, donor in part
                        ]
                    )
                    patched = _patched_query_batch(
                        model, tokenizer, queries, layer, anchor, vectors
                    )
                    for (recipient, donor), legal_logits in zip(part, patched, strict=True):
                        observed_recipient = behavior_rows[recipient["condition_id"]]
                        observed_donor = behavior_rows[donor["condition_id"]]
                        destination_margin = correct_action_margin(
                            recipient, observed_recipient["action_result"]["legal_logits"]
                        )
                        donor_margin = correct_action_margin(
                            donor, observed_donor["action_result"]["legal_logits"]
                        )
                        patched_margin = correct_action_margin(recipient, legal_logits)
                        output.append(
                            {
                                "recipient_condition_id": recipient["condition_id"],
                                "donor_condition_id": donor["condition_id"],
                                "base_game_id": recipient["base_game_id"],
                                "frame": recipient["frame"],
                                "donor_family": donor_family,
                                "layer": layer,
                                "anchor": anchor,
                                "destination_margin": destination_margin,
                                "donor_margin": donor_margin,
                                "patched_margin": patched_margin,
                                "margin_change": patched_margin - destination_margin,
                                "patched_choice": max(legal_logits, key=legal_logits.get),
                                "repaired": max(legal_logits, key=legal_logits.get)
                                == recipient["expected_action"],
                            }
                        )
    return output


def _capture_query_vectors(
    model: Any,
    tokenizer: Any,
    queries: list[dict[str, Any]],
    layer_index: int,
    anchor: str,
) -> Any:
    import numpy as np
    import torch

    input_ids, attention, pads = _left_padded(queries, tokenizer.pad_token_id)
    positions = torch.tensor(
        [
            pads[index] + query["anchor_positions"][anchor]
            for index, query in enumerate(queries)
        ],
        dtype=torch.long,
        device="cuda",
    )
    batch_indices = torch.arange(len(queries), device="cuda")
    captured: dict[str, Any] = {}

    def capture(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        captured["vectors"] = hidden[batch_indices, positions].detach().float().cpu().numpy()
        return output

    handle = _model_layers(model)[layer_index].register_forward_hook(capture)
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, attention_mask=attention, logits_to_keep=1)
    finally:
        handle.remove()
    del input_ids, attention
    return np.asarray(captured["vectors"], dtype=np.float16)


def _evaluate_locked_probes(
    residual_path: Path,
    mask_path: Path,
    metadata: list[dict[str, Any]],
    probe_artifact: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    import pickle

    import numpy as np
    from sklearn.metrics import balanced_accuracy_score

    if _sha256_file(Path(probe_artifact["remote_model_path"])) != probe_artifact[
        "remote_model_sha256"
    ]:
        raise RuntimeError("remote probe model hash mismatch")
    with gzip.open(probe_artifact["remote_model_path"], "rb") as handle:
        models = pickle.load(handle)
    features = np.load(residual_path, mmap_mode="r")
    masks = np.load(mask_path)
    rng = np.random.default_rng(int(config["probe"]["seed"]) + 1)
    resamples = int(config["probe"]["bootstrap_base_resamples"])
    output = []
    for record in models:
        anchor_index = int(record["anchor_index"])
        indices = np.flatnonzero(masks[:, anchor_index])
        if (
            record["model_kind"] in {"successful_formats", "donor_formats"}
            and config.get("probe", {}).get("evaluation_population", "prose") == "prose"
        ):
            indices = np.asarray(
                [
                    index
                    for index in indices
                    if metadata[index]["surface_kind"] == "assertion"
                    and metadata[index]["mapping_format"] == "prose"
                ],
                dtype=int,
            )
        x = _decode_numpy_features(
            features[indices, int(record["layer"]), anchor_index]
        )
        y = np.asarray([int(metadata[index][record["target"]]) for index in indices])
        predicted = record["estimator"].predict(x)
        point = float(balanced_accuracy_score(y, predicted))
        by_base: dict[str, list[int]] = {}
        for local_index, source_index in enumerate(indices):
            by_base.setdefault(metadata[source_index]["base_game_id"], []).append(local_index)
        bases = sorted(by_base)
        boot = []
        if len(set(y)) >= 2 and bases:
            for _ in range(resamples):
                sampled = rng.choice(bases, size=len(bases), replace=True)
                sampled_indices = [index for base in sampled for index in by_base[base]]
                sampled_y = y[sampled_indices]
                if len(set(sampled_y)) >= 2:
                    boot.append(
                        balanced_accuracy_score(sampled_y, predicted[sampled_indices])
                    )
        output.append(
            {
                "layer": int(record["layer"]),
                "anchor": record["anchor"],
                "target": record["target"],
                "model_kind": record["model_kind"],
                "n": len(indices),
                "balanced_accuracy": point,
                "bootstrap_95_low": float(np.quantile(boot, 0.025)) if boot else None,
                "bootstrap_95_high": float(np.quantile(boot, 0.975)) if boot else None,
            }
        )
    return output


def _run_locked_patches(
    model: Any,
    tokenizer: Any,
    dataset_rows: list[dict[str, Any]],
    behavior_rows: dict[str, dict[str, Any]],
    residual_path: Path,
    metadata: list[dict[str, Any]],
    anchor_names: list[str],
    patch_artifact: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    import numpy as np

    from jspace_policy.mechanistic_decomposition import (
        correct_action_margin,
        normalized_recovery,
    )
    from jspace_policy.mechanistic_decomposition_games import matched_row

    selected = patch_artifact["selected"]
    if selected is None:
        raise RuntimeError("locked patching requires a positive frozen discovery site")
    donor_family = selected["donor_family"]
    layer = int(selected["layer"])
    anchor = selected["anchor"]
    anchor_index = anchor_names.index(anchor)
    residuals = np.load(residual_path, mmap_mode="r")
    feature_index = {row["condition_id"]: index for index, row in enumerate(metadata)}
    locked_rows = [row for row in dataset_rows if row["split"] == "locked"]
    recipients = [
        row
        for row in locked_rows
        if row["incentive"] == "opposed"
        and row["surface_kind"] == "assertion"
        and row["history"] == "redundant"
        and row["mapping_format"] == "prose"
        and not behavior_rows[row["condition_id"]]["action_correct"]
        and all(
            report["correct"]
            for report in behavior_rows[row["condition_id"]]["option_reports"].values()
        )
    ]
    pairs = []
    for recipient in recipients:
        donor = matched_row(locked_rows, recipient, donor_family)
        if behavior_rows[donor["condition_id"]]["action_correct"]:
            pairs.append((recipient, donor))
    output: dict[str, Any] = {"primary": [], "controls": [], "reports": []}
    batch_size = int(config["execution"]["patch_batch_size"])
    for start in range(0, len(pairs), batch_size):
        part = pairs[start : start + batch_size]
        queries = [_action_query(tokenizer, recipient) for recipient, _ in part]
        donor_vectors = np.stack(
            [
                residuals[feature_index[donor["condition_id"]], layer, anchor_index]
                for _, donor in part
            ]
        )
        patched = _patched_query_batch(
            model, tokenizer, queries, layer, anchor, donor_vectors
        )
        identity_vectors = np.stack(
            [
                residuals[
                    feature_index[recipient["condition_id"]], layer, anchor_index
                ]
                for recipient, _ in part
            ]
        )
        identities = _patched_query_batch(
            model, tokenizer, queries, layer, anchor, identity_vectors
        )
        for (recipient, donor), legal_logits, identity_logits in zip(
            part, patched, identities, strict=True
        ):
            destination = behavior_rows[recipient["condition_id"]]
            source = behavior_rows[donor["condition_id"]]
            destination_margin = correct_action_margin(
                recipient, destination["action_result"]["legal_logits"]
            )
            donor_margin = correct_action_margin(
                donor, source["action_result"]["legal_logits"]
            )
            patched_margin = correct_action_margin(recipient, legal_logits)
            output["primary"].append(
                {
                    "recipient_condition_id": recipient["condition_id"],
                    "donor_condition_id": donor["condition_id"],
                    "base_game_id": recipient["base_game_id"],
                    "frame": recipient["frame"],
                    "destination_margin": destination_margin,
                    "donor_margin": donor_margin,
                    "patched_margin": patched_margin,
                    "margin_change": patched_margin - destination_margin,
                    "normalized_recovery": normalized_recovery(
                        destination_margin,
                        donor_margin,
                        patched_margin,
                        float(config["patch"]["normalized_recovery_minimum_denominator"]),
                    ),
                    "patched_choice": max(legal_logits, key=legal_logits.get),
                    "repaired": max(legal_logits, key=legal_logits.get)
                    == recipient["expected_action"],
                }
            )
            identity_margin = correct_action_margin(recipient, identity_logits)
            output["controls"].append(
                {
                    "control": "identity",
                    "condition_id": recipient["condition_id"],
                    "base_game_id": recipient["base_game_id"],
                    "margin_change": identity_margin - destination_margin,
                    "choice_changed": max(identity_logits, key=identity_logits.get)
                    != destination["selected_action"],
                }
            )

        # Reverse direction: failed prose states inserted into successful donors.
        reverse_queries = [_action_query(tokenizer, donor) for _, donor in part]
        reverse_vectors = identity_vectors
        reversed_logits = _patched_query_batch(
            model, tokenizer, reverse_queries, layer, anchor, reverse_vectors
        )
        for (recipient, donor), legal_logits in zip(part, reversed_logits, strict=True):
            destination = behavior_rows[donor["condition_id"]]
            baseline_margin = correct_action_margin(
                donor, destination["action_result"]["legal_logits"]
            )
            patched_margin = correct_action_margin(donor, legal_logits)
            output["controls"].append(
                {
                    "control": "reverse_prose_into_success",
                    "condition_id": donor["condition_id"],
                    "source_condition_id": recipient["condition_id"],
                    "base_game_id": donor["base_game_id"],
                    "margin_change": patched_margin - baseline_margin,
                    "correct_after": max(legal_logits, key=legal_logits.get)
                    == donor["expected_action"],
                }
            )

        # Opposite-target states come from the same-base aligned prose condition.
        opposite_pairs = []
        for recipient, _ in part:
            aligned = next(
                row
                for row in locked_rows
                if row["base_game_id"] == recipient["base_game_id"]
                and row["frame"] == recipient["frame"]
                and row["incentive"] == "aligned"
            )
            opposite_pairs.append((recipient, aligned))
        opposite_vectors = np.stack(
            [
                residuals[feature_index[source["condition_id"]], layer, anchor_index]
                for _, source in opposite_pairs
            ]
        )
        opposite_logits = _patched_query_batch(
            model, tokenizer, queries, layer, anchor, opposite_vectors
        )
        for (recipient, source), legal_logits in zip(
            opposite_pairs, opposite_logits, strict=True
        ):
            destination = behavior_rows[recipient["condition_id"]]
            baseline_margin = correct_action_margin(
                recipient, destination["action_result"]["legal_logits"]
            )
            output["controls"].append(
                {
                    "control": "opposite_target_same_base",
                    "condition_id": recipient["condition_id"],
                    "source_condition_id": source["condition_id"],
                    "base_game_id": recipient["base_game_id"],
                    "margin_change": correct_action_margin(recipient, legal_logits)
                    - baseline_margin,
                    "repaired": max(legal_logits, key=legal_logits.get)
                    == recipient["expected_action"],
                }
            )

    # Same-condition cross-base control, matched on frame and correct action.
    successful_prose = [
        row
        for row in locked_rows
        if row["incentive"] == "opposed"
        and row["surface_kind"] == "assertion"
        and row["history"] == "redundant"
        and row["mapping_format"] == "prose"
        and behavior_rows[row["condition_id"]]["action_correct"]
    ]
    cross_pairs = []
    for recipient, _ in pairs:
        candidates = [
            row
            for row in successful_prose
            if row["base_game_id"] != recipient["base_game_id"]
            and row["frame"] == recipient["frame"]
            and row["expected_action"] == recipient["expected_action"]
        ]
        if candidates:
            source = sorted(candidates, key=lambda row: row["condition_id"])[0]
            cross_pairs.append((recipient, source))
    for start in range(0, len(cross_pairs), batch_size):
        part = cross_pairs[start : start + batch_size]
        queries = [_action_query(tokenizer, recipient) for recipient, _ in part]
        vectors = np.stack(
            [
                residuals[feature_index[source["condition_id"]], layer, anchor_index]
                for _, source in part
            ]
        )
        logits = _patched_query_batch(model, tokenizer, queries, layer, anchor, vectors)
        for (recipient, source), legal_logits in zip(part, logits, strict=True):
            destination = behavior_rows[recipient["condition_id"]]
            baseline_margin = correct_action_margin(
                recipient, destination["action_result"]["legal_logits"]
            )
            output["controls"].append(
                {
                    "control": "same_condition_cross_base",
                    "condition_id": recipient["condition_id"],
                    "source_condition_id": source["condition_id"],
                    "base_game_id": recipient["base_game_id"],
                    "margin_change": correct_action_margin(recipient, legal_logits)
                    - baseline_margin,
                    "repaired": max(legal_logits, key=legal_logits.get)
                    == recipient["expected_action"],
                }
            )

    # Non-damage controls use natural same-condition states from another base.
    control_categories = {
        "aligned": lambda row: row["incentive"] == "aligned",
        "table": lambda row: row["incentive"] == "opposed"
        and row["surface_kind"] == "assertion"
        and row["mapping_format"] == "table"
        and row["history"] == "redundant",
        "opaque": lambda row: row["incentive"] == "opposed"
        and row["surface_kind"] == "opaque_token"
        and row["history"] == "redundant",
    }
    non_damage_pairs = []
    for category, predicate in control_categories.items():
        destinations = [
            row
            for row in locked_rows
            if predicate(row) and behavior_rows[row["condition_id"]]["action_correct"]
        ]
        for destination in destinations:
            candidates = [
                row
                for row in destinations
                if row["base_game_id"] != destination["base_game_id"]
                and row["frame"] == destination["frame"]
                and row["expected_action"] == destination["expected_action"]
            ]
            if candidates:
                source = sorted(candidates, key=lambda row: row["condition_id"])[0]
                non_damage_pairs.append((category, destination, source))
    for start in range(0, len(non_damage_pairs), batch_size):
        part = non_damage_pairs[start : start + batch_size]
        queries = [_action_query(tokenizer, destination) for _, destination, _ in part]
        vectors = np.stack(
            [
                residuals[feature_index[source["condition_id"]], layer, anchor_index]
                for _, _, source in part
            ]
        )
        logits = _patched_query_batch(model, tokenizer, queries, layer, anchor, vectors)
        for (category, destination, source), legal_logits in zip(part, logits, strict=True):
            observed = behavior_rows[destination["condition_id"]]
            baseline_margin = correct_action_margin(
                destination, observed["action_result"]["legal_logits"]
            )
            output["controls"].append(
                {
                    "control": f"non_damage_{category}",
                    "condition_id": destination["condition_id"],
                    "source_condition_id": source["condition_id"],
                    "base_game_id": destination["base_game_id"],
                    "margin_change": correct_action_margin(destination, legal_logits)
                    - baseline_margin,
                    "correct_before": True,
                    "correct_after": max(legal_logits, key=legal_logits.get)
                    == destination["expected_action"],
                }
            )

    # Apply the same natural source/destination pairing to both consequence reports.
    report_pairs = [
        (recipient, donor, option)
        for recipient, donor in pairs
        for option in ("A", "B")
    ]
    for start in range(0, len(report_pairs), batch_size):
        part = report_pairs[start : start + batch_size]
        source_queries = [
            _report_query(tokenizer, donor, option)[0] for _, donor, option in part
        ]
        destination_queries = [
            _report_query(tokenizer, recipient, option)[0]
            for recipient, _, option in part
        ]
        vectors = _capture_query_vectors(
            model, tokenizer, source_queries, layer, anchor
        )
        logits = _patched_query_batch(
            model, tokenizer, destination_queries, layer, anchor, vectors
        )
        for (recipient, donor, option), legal_logits in zip(part, logits, strict=True):
            expected = behavior_rows[recipient["condition_id"]]["option_reports"][option][
                "expected"
            ]
            output["reports"].append(
                {
                    "recipient_condition_id": recipient["condition_id"],
                    "donor_condition_id": donor["condition_id"],
                    "base_game_id": recipient["base_game_id"],
                    "option": option,
                    "expected": expected,
                    "patched_choice": max(legal_logits, key=legal_logits.get),
                    "correct_before": True,
                    "correct_after": max(legal_logits, key=legal_logits.get) == expected,
                }
            )
    return output


@app.function(
    image=core_image,
    gpu="A100-80GB",
    cpu=16,
    memory=65536,
    volumes={"/cache": cache, "/artifacts": artifacts},
    timeout=7200,
    retries=0,
    max_containers=1,
)
def discovery_remote(
    dataset: dict[str, Any],
    behavior_payload: dict[str, Any],
    config: dict[str, Any],
    git_commit: str,
) -> str:
    import torch

    from jspace_policy.mechanistic_decomposition import select_patch_site
    from jspace_policy.mechanistic_decomposition_analysis import compute_activation_geometry

    _validate(config, dataset)
    if behavior_payload["metadata"]["dataset_sha256"] != dataset["content_sha256"]:
        raise RuntimeError("behavior/dataset hash mismatch")
    if not behavior_payload["analysis"]["gate_pass"]:
        raise RuntimeError("behavioral gate failed; discovery activations remain unopened")
    torch.manual_seed(int(config["execution"]["seed"]))
    started = time.perf_counter()
    model, tokenizer, model_metadata = _load_model()
    run_id = uuid.uuid4().hex
    remote_root = Path(f"/artifacts/{_artifact_prefix(config)}/{run_id}")
    remote_root.mkdir(parents=True, exist_ok=False)
    behavior_rows = {row["condition_id"]: row for row in behavior_payload["rows"]}
    selected_rows = [
        row for row in dataset["rows"]
        if row["split"] == "discovery" and _capture_row_allowed(row, config)
    ]
    residual_path = remote_root / "discovery_residuals.npy"
    metadata, anchor_names, layers = _capture_split(
        model,
        tokenizer,
        selected_rows,
        behavior_rows,
        config,
        residual_path,
    )
    metadata_path = remote_root / "discovery_metadata.json.gz"
    with gzip.open(metadata_path, "wt", encoding="utf-8") as handle:
        json.dump(metadata, handle, sort_keys=True)
    geometry_path = remote_root / "discovery_geometry.json.gz"
    geometry_artifact = compute_activation_geometry(
        residual_path,
        metadata,
        anchor_names,
        layers,
        dataset["rows"],
        geometry_path,
        permutation_seed=int(config["execution"]["seed"]) + 2,
    )
    probe_artifact = _fit_probe_artifact(
        residual_path,
        residual_path.with_name("anchor_mask.npy"),
        metadata,
        anchor_names,
        layers,
        config,
        remote_root,
        dataset["content_sha256"],
    )
    patch_rows = _run_discovery_patches(
        model,
        tokenizer,
        dataset["rows"],
        behavior_rows,
        residual_path,
        metadata,
        anchor_names,
        config,
    )
    patch_path = remote_root / "discovery_patches.json.gz"
    with gzip.open(patch_path, "wt", encoding="utf-8") as handle:
        json.dump(patch_rows, handle, sort_keys=True)
    patch_artifact = select_patch_site(
        patch_rows, config, str(dataset["content_sha256"])
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    artifacts.commit()
    cache.commit()
    result = {
        "schema_version": 1,
        "study_id": _study_id(config),
        "metadata": {
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": elapsed,
            "git_commit": git_commit,
            "phase": "discovery",
            "dataset_sha256": dataset["content_sha256"],
            "config_sha256": _canonical_sha256(config),
            **model_metadata,
        },
        "artifact": {
            "remote_root": str(remote_root),
            "residual_path": str(residual_path),
            "residual_sha256": _sha256_file(residual_path),
            "residual_shape": [
                len(metadata), len(layers), len(anchor_names), int(model_metadata["d_model"])
            ],
            "residual_storage_dtype": config["capture"].get("storage_dtype", "float16"),
            "anchor_mask_path": str(residual_path.with_name("anchor_mask.npy")),
            "anchor_mask_sha256": _sha256_file(
                residual_path.with_name("anchor_mask.npy")
            ),
            "metadata_path": str(metadata_path),
            "metadata_sha256": _sha256_file(metadata_path),
            "geometry_path": str(geometry_path),
            "geometry_sha256": geometry_artifact["sha256"],
            "patch_rows_path": str(patch_path),
            "patch_rows_sha256": _sha256_file(patch_path),
            "anchor_names": anchor_names,
            "layers": layers,
        },
        "probe_freeze": probe_artifact,
        "patch_freeze": patch_artifact,
    }
    return json.dumps(result, allow_nan=False, sort_keys=True)


@app.function(
    image=core_image,
    gpu="A100-80GB",
    cpu=16,
    memory=65536,
    volumes={"/cache": cache, "/artifacts": artifacts},
    timeout=9000,
    retries=0,
    max_containers=1,
)
def locked_remote(
    dataset: dict[str, Any],
    behavior_payload: dict[str, Any],
    discovery_manifest: dict[str, Any],
    config: dict[str, Any],
    git_commit: str,
) -> str:
    import torch

    _validate(config, dataset)
    if not behavior_payload["analysis"]["gate_pass"]:
        raise RuntimeError("behavioral gate failed; locked activations remain unopened")
    probe_artifact = discovery_manifest["probe_freeze"]
    patch_artifact = discovery_manifest["patch_freeze"]
    if not _content_hash_valid(probe_artifact) or not _content_hash_valid(patch_artifact):
        raise RuntimeError("invalid discovery freeze hash")
    if patch_artifact["status"] != "frozen_after_discovery_before_locked_capture":
        raise RuntimeError("no positive discovery patch; locked phase remains closed")
    if probe_artifact["dataset_sha256"] != dataset["content_sha256"]:
        raise RuntimeError("probe freeze/dataset mismatch")
    torch.manual_seed(int(config["execution"]["seed"]))
    started = time.perf_counter()
    model, tokenizer, model_metadata = _load_model()
    run_id = uuid.uuid4().hex
    remote_root = Path(f"/artifacts/{_artifact_prefix(config)}/{run_id}")
    remote_root.mkdir(parents=True, exist_ok=False)
    behavior_rows = {row["condition_id"]: row for row in behavior_payload["rows"]}
    selected_rows = [
        row for row in dataset["rows"]
        if row["split"] == "locked" and _capture_row_allowed(row, config)
    ]
    residual_path = remote_root / "locked_residuals.npy"
    metadata, anchor_names, layers = _capture_split(
        model,
        tokenizer,
        selected_rows,
        behavior_rows,
        config,
        residual_path,
    )
    metadata_path = remote_root / "locked_metadata.json.gz"
    with gzip.open(metadata_path, "wt", encoding="utf-8") as handle:
        json.dump(metadata, handle, sort_keys=True)
    from jspace_policy.mechanistic_decomposition_analysis import compute_activation_geometry

    geometry_path = remote_root / "locked_geometry.json.gz"
    geometry_artifact = compute_activation_geometry(
        residual_path,
        metadata,
        anchor_names,
        layers,
        dataset["rows"],
        geometry_path,
        permutation_seed=int(config["execution"]["seed"]) + 3,
    )
    probe_metrics = _evaluate_locked_probes(
        residual_path,
        residual_path.with_name("anchor_mask.npy"),
        metadata,
        probe_artifact,
        config,
    )
    probe_metrics_path = remote_root / "locked_probe_metrics.json.gz"
    with gzip.open(probe_metrics_path, "wt", encoding="utf-8") as handle:
        json.dump(probe_metrics, handle, sort_keys=True)
    patch_results = _run_locked_patches(
        model,
        tokenizer,
        dataset["rows"],
        behavior_rows,
        residual_path,
        metadata,
        anchor_names,
        patch_artifact,
        config,
    )
    patch_results_path = remote_root / "locked_patches.json.gz"
    with gzip.open(patch_results_path, "wt", encoding="utf-8") as handle:
        json.dump(patch_results, handle, sort_keys=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    artifacts.commit()
    cache.commit()
    result = {
        "schema_version": 1,
        "study_id": _study_id(config),
        "metadata": {
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": elapsed,
            "git_commit": git_commit,
            "phase": "locked",
            "dataset_sha256": dataset["content_sha256"],
            "config_sha256": _canonical_sha256(config),
            "probe_freeze_sha256": probe_artifact["content_sha256"],
            "patch_freeze_sha256": patch_artifact["content_sha256"],
            **model_metadata,
        },
        "artifact": {
            "remote_root": str(remote_root),
            "residual_path": str(residual_path),
            "residual_sha256": _sha256_file(residual_path),
            "residual_shape": [
                len(metadata), len(layers), len(anchor_names), int(model_metadata["d_model"])
            ],
            "residual_storage_dtype": config["capture"].get("storage_dtype", "float16"),
            "anchor_mask_path": str(residual_path.with_name("anchor_mask.npy")),
            "anchor_mask_sha256": _sha256_file(
                residual_path.with_name("anchor_mask.npy")
            ),
            "metadata_path": str(metadata_path),
            "metadata_sha256": _sha256_file(metadata_path),
            "geometry_path": str(geometry_path),
            "geometry_sha256": geometry_artifact["sha256"],
            "probe_metrics_path": str(probe_metrics_path),
            "probe_metrics_sha256": _sha256_file(probe_metrics_path),
            "patch_results_path": str(patch_results_path),
            "patch_results_sha256": _sha256_file(patch_results_path),
            "anchor_names": anchor_names,
            "layers": layers,
        },
        "counts": {
            "probe_metrics": len(probe_metrics),
            "primary_patch_rows": len(patch_results["primary"]),
            "control_patch_rows": len(patch_results["controls"]),
            "report_patch_rows": len(patch_results["reports"]),
        },
    }
    return json.dumps(result, allow_nan=False, sort_keys=True)


@app.function(
    image=lens_image,
    gpu="A100-80GB",
    cpu=8,
    memory=65536,
    volumes={"/cache": cache, "/artifacts": artifacts},
    timeout=5400,
    retries=0,
    max_containers=1,
)
def jspace_remote(
    dataset: dict[str, Any],
    discovery_manifest: dict[str, Any],
    locked_manifest: dict[str, Any],
    config: dict[str, Any],
    git_commit: str,
) -> str:
    import numpy as np
    import torch
    from jlens import JacobianLens, from_hf

    _validate(config, dataset)
    started = time.perf_counter()
    hf_model, tokenizer, model_metadata = _load_model()
    model = from_hf(hf_model, tokenizer)
    lens = JacobianLens.from_pretrained(
        config["jspace"]["lens_repo"],
        filename=config["jspace"]["lens_filename"],
        revision=config["jspace"]["lens_revision"],
    )
    if int(lens.d_model) != int(model.d_model):
        raise RuntimeError("pinned Jacobian Lens residual width mismatch")
    if config["jspace"].get("candidate_score_all_rows", False):
        # Verify the optimized candidate-token projection against the lens wrapper's
        # full unembedding before using it for the large all-row trajectory.
        with torch.inference_mode():
            probe_state = torch.zeros((1, int(model.d_model)), device="cuda")
            probe_ids = [0, 1, 2]
            full_probe = model.unembed(probe_state).float()[0, probe_ids]
            direct_probe = (
                probe_state @ hf_model.lm_head.weight[probe_ids].to(dtype=probe_state.dtype).T
            ).float()[0]
            if getattr(hf_model.lm_head, "bias", None) is not None:
                direct_probe += hf_model.lm_head.bias[probe_ids].float().to(device="cuda")
            if not torch.allclose(full_probe, direct_probe, atol=1e-4, rtol=1e-5):
                raise RuntimeError(
                    "optimized J-space projection failed full-unembedding parity"
                )
    run_id = uuid.uuid4().hex
    remote_root = Path(f"/artifacts/{_artifact_prefix(config)}/{run_id}")
    remote_root.mkdir(parents=True, exist_ok=False)
    output = []
    by_condition = {row["condition_id"]: row for row in dataset["rows"]}
    for split, manifest in (
        ("discovery", discovery_manifest),
        ("locked", locked_manifest),
    ):
        artifact = manifest["artifact"]
        residual_path = Path(artifact["residual_path"])
        metadata_path = Path(artifact["metadata_path"])
        if _sha256_file(residual_path) != artifact["residual_sha256"]:
            raise RuntimeError(f"{split} residual hash mismatch")
        if _sha256_file(metadata_path) != artifact["metadata_sha256"]:
            raise RuntimeError(f"{split} metadata hash mismatch")
        residuals = np.load(residual_path, mmap_mode="r")
        with gzip.open(metadata_path, "rt", encoding="utf-8") as handle:
            metadata = json.load(handle)
        base_ids = sorted({row["base_game_id"] for row in metadata})
        if config["jspace"].get("candidate_score_all_rows", False):
            bases = base_ids
        else:
            bases = base_ids[: int(config["jspace"].get("bases_per_split", 4))]
        full_vocab_bases = set(bases)
        if config["jspace"].get("candidate_score_all_rows", False):
            n_full = int(config["jspace"].get("full_vocab_bases_per_split", 0))
            seed = int(config.get("execution", {}).get("seed", 0))
            full_vocab_bases = {
                base
                for _, base in sorted(
                    (
                        hashlib.sha256(f"{seed}:{split}:{base}".encode()).hexdigest(),
                        base,
                    )
                    for base in base_ids
                )[:n_full]
            }
        indices = [
            index for index, row in enumerate(metadata) if row["base_game_id"] in bases
        ]
        anchor_names = list(artifact["anchor_names"])
        for layer in map(int, config["jspace"]["layers"]):
            jacobian = lens.jacobians[layer].to("cuda")
            for anchor_index, anchor in enumerate(anchor_names):
                for start in range(0, len(indices), 8):
                    part = indices[start : start + 8]
                    states = _decode_residual_slice(
                        residuals[part, layer, anchor_index], device="cuda"
                    ).float()
                    with torch.inference_mode():
                        transported = states @ jacobian.T
                        rows_for_part = [metadata[index] for index in part]
                        top_enabled = any(
                            row["base_game_id"] in full_vocab_bases
                            and anchor in config["jspace"].get(
                                "full_vocab_anchors", anchor_names
                            )
                            for row in rows_for_part
                        )
                        if top_enabled:
                            logits = model.unembed(transported).float()
                            top = logits.topk(int(config["jspace"]["top_k"]), dim=-1)
                            selected_logits = None
                        else:
                            required_ids = sorted(
                                {
                                    token_id
                                    for source_index in part
                                    for token_id in (
                                        list(
                                            map(
                                                int,
                                                _action_query(
                                                    tokenizer,
                                                    by_condition[metadata[source_index]["condition_id"]],
                                                )["candidate_token_ids"],
                                            )
                                        )
                                        + [
                                            token_id
                                            for concept in by_condition[
                                                metadata[source_index]["condition_id"]
                                            ]["concepts"]
                                            for token_id in tokenizer.encode(
                                                f" {concept}", add_special_tokens=False
                                            )
                                        ]
                                    )
                                }
                            )
                            weight = hf_model.lm_head.weight[required_ids].to(
                                device=transported.device, dtype=transported.dtype
                            )
                            selected_logits = transported @ weight.T
                            if getattr(hf_model.lm_head, "bias", None) is not None:
                                selected_logits += hf_model.lm_head.bias[required_ids].to(
                                    device=transported.device, dtype=transported.dtype
                                )
                            logits = None
                            top = None
                    for local_index, source_index in enumerate(part):
                        item = metadata[source_index]
                        row = by_condition[item["condition_id"]]
                        query = _action_query(tokenizer, row)
                        full_vocab_row = (
                            row["base_game_id"] in full_vocab_bases
                            and anchor in config["jspace"].get(
                                "full_vocab_anchors", anchor_names
                            )
                        )
                        action_ids = list(map(int, query["candidate_token_ids"]))
                        if logits is not None:
                            action_scores = logits[local_index, action_ids]
                        else:
                            score_index = {
                                token_id: offset
                                for offset, token_id in enumerate(required_ids)
                            }
                            action_scores = torch.stack(
                                [
                                    selected_logits[local_index, score_index[token_id]]
                                    for token_id in action_ids
                                ]
                            )
                        concept_scores = {}
                        for concept in row["concepts"]:
                            concept_ids = tokenizer.encode(
                                f" {concept}", add_special_tokens=False
                            )
                            if logits is not None:
                                concept_scores[concept] = float(
                                    logits[local_index, concept_ids].mean().cpu()
                                )
                            else:
                                concept_scores[concept] = float(
                                    torch.stack(
                                        [
                                            selected_logits[local_index, score_index[token_id]]
                                            for token_id in concept_ids
                                        ]
                                    ).mean().cpu()
                                )
                        output.append(
                            {
                                "split": split,
                                "condition_id": row["condition_id"],
                                "base_game_id": row["base_game_id"],
                                "layer": layer,
                                "anchor": anchor,
                                "action_scores": dict(
                                    zip(
                                        ["A", "B"],
                                        map(float, action_scores.cpu()),
                                        strict=True,
                                    )
                                ),
                                "concept_mean_token_scores": concept_scores,
                                "target_concept": row["target_response"],
                                "top_token_ids": (
                                    list(map(int, top.indices[local_index].cpu()))
                                    if top is not None and full_vocab_row
                                    else []
                                ),
                                "top_tokens": (
                                    [
                                        tokenizer.decode([int(token)])
                                        for token in top.indices[local_index].cpu()
                                    ]
                                    if top is not None and full_vocab_row
                                    else []
                                ),
                                "top_scores": (
                                    list(map(float, top.values[local_index].cpu()))
                                    if top is not None and full_vocab_row
                                    else []
                                ),
                            }
                        )
                    del states, transported, logits, top
            del jacobian
    output_path = remote_root / "jspace_rows.json.gz"
    with gzip.open(output_path, "wt", encoding="utf-8") as handle:
        json.dump(output, handle, sort_keys=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    artifacts.commit()
    cache.commit()
    return json.dumps(
        {
            "schema_version": 1,
            "study_id": _study_id(config),
            "status": "completed_observational_secondary",
            "metadata": {
                "run_id": run_id,
                "created_at": datetime.now(UTC).isoformat(),
                "elapsed_seconds": elapsed,
                "git_commit": git_commit,
                "dataset_sha256": dataset["content_sha256"],
                "lens_repo": config["jspace"]["lens_repo"],
                "lens_revision": config["jspace"]["lens_revision"],
                "lens_filename": config["jspace"]["lens_filename"],
                "lens_code_commit": config["jspace"]["lens_code_commit"],
                **model_metadata,
            },
            "artifact": {
                "remote_root": str(remote_root),
                "rows_path": str(output_path),
                "rows_sha256": _sha256_file(output_path),
                "n_rows": len(output),
            },
        },
        allow_nan=False,
        sort_keys=True,
    )


def _v5_buffered_total() -> float:
    return sum(ledger_total(path) for path in Path("results").glob("v5*/cost_ledger.jsonl"))


def _stage_estimate(config: dict[str, Any], stage: str) -> Any:
    seconds_key = {
        "behavior": "estimated_behavior_ceiling_seconds",
        "discovery": "estimated_capture_ceiling_seconds",
        "locked": "estimated_patch_ceiling_seconds",
        "jspace": "estimated_jspace_ceiling_seconds",
    }[stage]
    memory = 32 if stage == "behavior" else 64
    cpu = 8 if stage in {"behavior", "jspace"} else 16
    if seconds_key not in config["execution"]:
        seconds_key = "estimated_capture_ceiling_seconds"
    return estimate_cost(
        str(config["execution"]["gpu"]),
        float(config["execution"][seconds_key]),
        cpu_cores=cpu,
        memory_gib=memory,
    )


def _admit_stage(config: dict[str, Any], stage: str) -> Any:
    estimate = _stage_estimate(config, stage)
    projected = _v5_buffered_total() + estimate.buffered_usd
    limit = float(config["execution"]["hard_cumulative_v5_cost_limit_usd"])
    if projected > limit:
        raise RuntimeError(
            f"{stage} refused: cumulative buffered V5 cost would reach "
            f"${projected:.2f}, above ${limit:.2f}"
        )
    return estimate


def _record_stage(config: dict[str, Any], payload: dict[str, Any], stage: str) -> None:
    memory = 32 if stage == "behavior" else 64
    cpu = 8 if stage in {"behavior", "jspace"} else 16
    measured = estimate_cost(
        str(config["execution"]["gpu"]),
        float(payload["metadata"]["elapsed_seconds"]),
        cpu_cores=cpu,
        memory_gib=memory,
    )
    append_ledger(
        _result_root(config) / "cost_ledger.jsonl",
        measured,
        run_id=str(payload["metadata"]["run_id"]),
        stage=f"{_artifact_prefix(config)}_{stage}",
    )


@app.local_entrypoint()
def freeze_dataset() -> None:
    from jspace_policy.mechanistic_decomposition_games import (
        dataset_payload,
        verify_dataset_payload,
    )

    config = _load_json(CONFIG_PATH)
    payload = dataset_payload(config)
    verify_dataset_payload(payload, config)
    manifest = _load_json(MANIFEST_PATH)
    if payload["content_sha256"] != manifest["expected_content_sha256"]:
        raise RuntimeError("generated dataset differs from prospective manifest")
    _write_new(DATASET_PATH, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(payload["rows"]), "sha256": payload["content_sha256"]}))


@app.local_entrypoint()
def preflight() -> None:
    config = _load_json(CONFIG_PATH)
    dataset = _load_json(DATASET_PATH)
    _validate(config, dataset)
    payload = json.loads(preflight_remote.remote(dataset, config))
    _write_new(
        RESULT_ROOT / "raw/preflight.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


@app.local_entrypoint()
def behavior() -> None:
    if not _tracked_tree_clean():
        raise RuntimeError("behavior execution requires a clean tracked worktree")
    config = _load_json(CONFIG_PATH)
    dataset = _load_json(DATASET_PATH)
    _validate(config, dataset)
    _admit_stage(config, "behavior")
    payload = json.loads(behavior_remote.remote(dataset, config, _git_head()))
    _write_new(
        RESULT_ROOT / "raw/behavior.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    _write_new(
        RESULT_ROOT / "analysis/behavior_analysis.json",
        json.dumps(payload["analysis"], indent=2, sort_keys=True) + "\n",
    )
    _record_stage(config, payload, "behavior")
    print(json.dumps(payload["analysis"], indent=2, sort_keys=True))


@app.local_entrypoint()
def discovery() -> None:
    if not _tracked_tree_clean():
        raise RuntimeError("discovery execution requires a clean tracked worktree")
    config = _load_json(CONFIG_PATH)
    dataset = _load_json(DATASET_PATH)
    behavior_payload = _load_json(RESULT_ROOT / "raw/behavior.json")
    _validate(config, dataset)
    _admit_stage(config, "discovery")
    payload = json.loads(
        discovery_remote.remote(dataset, behavior_payload, config, _git_head())
    )
    _write_new(
        RESULT_ROOT / "raw/discovery_manifest.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    _write_new(
        RESULT_ROOT / "probe_freeze.json",
        json.dumps(payload["probe_freeze"], indent=2, sort_keys=True) + "\n",
    )
    _write_new(
        RESULT_ROOT / "patch_freeze.json",
        json.dumps(payload["patch_freeze"], indent=2, sort_keys=True) + "\n",
    )
    _record_stage(config, payload, "discovery")
    print(json.dumps(payload["patch_freeze"], indent=2, sort_keys=True))


@app.local_entrypoint()
def locked() -> None:
    if not _tracked_tree_clean():
        raise RuntimeError("locked execution requires a clean tracked worktree")
    config = _load_json(CONFIG_PATH)
    dataset = _load_json(DATASET_PATH)
    behavior_payload = _load_json(RESULT_ROOT / "raw/behavior.json")
    discovery_manifest = _load_json(RESULT_ROOT / "raw/discovery_manifest.json")
    _validate(config, dataset)
    _admit_stage(config, "locked")
    payload = json.loads(
        locked_remote.remote(
            dataset,
            behavior_payload,
            discovery_manifest,
            config,
            _git_head(),
        )
    )
    _write_new(
        RESULT_ROOT / "raw/locked_manifest.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    _record_stage(config, payload, "locked")
    print(json.dumps(payload["counts"], indent=2, sort_keys=True))


@app.local_entrypoint()
def jspace() -> None:
    config = _load_json(CONFIG_PATH)
    dataset = _load_json(DATASET_PATH)
    discovery_manifest = _load_json(RESULT_ROOT / "raw/discovery_manifest.json")
    locked_manifest = _load_json(RESULT_ROOT / "raw/locked_manifest.json")
    try:
        _admit_stage(config, "jspace")
    except RuntimeError as error:
        payload = {
            "schema_version": 1,
            "study_id": _study_id(config),
            "status": config["jspace"]["budget_skip_status"],
            "created_at": datetime.now(UTC).isoformat(),
            "reason": str(error),
        }
        _write_new(
            RESULT_ROOT / "raw/jspace_manifest.json",
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    payload = json.loads(
        jspace_remote.remote(
            dataset,
            discovery_manifest,
            locked_manifest,
            config,
            _git_head(),
        )
    )
    _write_new(
        RESULT_ROOT / "raw/jspace_manifest.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    _record_stage(config, payload, "jspace")
    print(json.dumps(payload["artifact"], indent=2, sort_keys=True))
