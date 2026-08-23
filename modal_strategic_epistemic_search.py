"""Budget-gated V4 strategic epistemic search.

Commands:
    modal run modal_strategic_epistemic_search.py::freeze_dataset
    modal run modal_strategic_epistemic_search.py::preflight
    modal run modal_strategic_epistemic_search.py::calibration_preflight
    modal run modal_strategic_epistemic_search.py::calibration
    modal run modal_strategic_epistemic_search.py::behavior
    modal run modal_strategic_epistemic_search.py::report_preflight
    modal run modal_strategic_epistemic_search.py::report
    modal run modal_strategic_epistemic_search.py::report_confirmation_preflight
    modal run modal_strategic_epistemic_search.py::report_confirmation
    modal run modal_strategic_epistemic_search.py::report_alias_preflight
    modal run modal_strategic_epistemic_search.py::report_alias
    modal run modal_strategic_epistemic_search.py::freeze_ordinal_dataset
    modal run modal_strategic_epistemic_search.py::ordinal_preflight
    modal run modal_strategic_epistemic_search.py::ordinal_behavior
    modal run modal_strategic_epistemic_search.py::ordinal_report
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
from fractions import Fraction
from pathlib import Path
from typing import Any

import modal

from jspace_policy.budget import admit_run, append_ledger, estimate_cost

CONFIG_PATH = Path("configs/v4/strategic_epistemic_search/experiment.json")
CALIBRATION_CONFIG_PATH = Path(
    "configs/v4/strategic_epistemic_search/calibration.json"
)
REPORT_CONFIRMATION_CONFIG_PATH = Path(
    "configs/v4/strategic_epistemic_search/report_confirmation.json"
)
REPORT_ALIAS_CONFIG_PATH = Path(
    "configs/v4/strategic_epistemic_search/report_alias_mechanism.json"
)
ORDINAL_CONFIG_PATH = Path(
    "configs/v4/strategic_epistemic_search/ordinal_binding_permutation.json"
)
DATASET_PATH = Path("configs/v4/strategic_epistemic_search/dataset.json")
ORDINAL_DATASET_PATH = Path(
    "configs/v4/strategic_epistemic_search/ordinal_binding_dataset.json"
)
RESULT_ROOT = Path("results/v4_strategic_epistemic_search")

MODEL_ID = "Qwen/Qwen3.6-27B"
MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
JLENS_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
LENS_FILENAME = (
    "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt"
)
DECISION_SYSTEM_PROMPT = (
    "Follow the output instructions exactly. Compute carefully and check the "
    "comparison before giving the final answer."
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
    report = config["self_report"]
    if report["candidate_labels"] != ["X", "Y", "Z"]:
        raise RuntimeError("self-report labels must remain disjoint from signals")
    if report["version"] == "disjoint_labels_v1":
        if set(report["query_types"]) != {
            "decisive_response",
            "predicted_response",
            "decision_margin",
        }:
            raise RuntimeError("discovery self-report query set changed")
        if set(report["access_conditions"]) != {
            "retrospective",
            "answer_only",
            "matched_trajectory",
            "ordinal_trajectory",
            "reconstruction",
        }:
            raise RuntimeError("discovery self-report access controls changed")
    elif report["version"] == "trajectory_interference_confirmation_v1":
        if report["splits"] != ["validation", "locked"]:
            raise RuntimeError("confirmation must use only held-out splits")
        if report["query_types"] != ["predicted_response"]:
            raise RuntimeError("confirmation query changed")
        if set(report["access_conditions"]) != {"retrospective", "answer_only"}:
            raise RuntimeError("confirmation access contrast changed")
        if report["eligible_selected_actions"] != ["C"]:
            raise RuntimeError("confirmation selected-action stratum changed")
    elif report["version"] == "response_alias_mechanism_v1":
        if report["splits"] != ["discovery", "validation", "locked"]:
            raise RuntimeError("alias mechanism split set changed")
        if report["query_types"] != ["predicted_response"]:
            raise RuntimeError("alias mechanism query changed")
        if set(report["access_conditions"]) != {"retrospective", "answer_only"}:
            raise RuntimeError("alias mechanism access contrast changed")
        if report["eligible_selected_actions"] != ["C"]:
            raise RuntimeError("alias mechanism selected-action stratum changed")
        if report["response_naming_conditions"] != ["indexed", "arbitrary_alias"]:
            raise RuntimeError("alias mechanism naming conditions changed")
        if report["response_aliases"] != ["Kestrel", "Lumen", "Quartz"]:
            raise RuntimeError("response aliases changed")
    elif report["version"] == "cross_role_ordinal_binding_v1":
        if config["dataset"].get("ordinal_binding_oa") != "OA(9,4,3,2)":
            raise RuntimeError("ordinal-binding design changed")
        if report["splits"] != ["locked"]:
            raise RuntimeError("ordinal-binding study must use its fresh locked split")
        if report["query_types"] != ["predicted_response"]:
            raise RuntimeError("ordinal-binding query changed")
        if set(report["access_conditions"]) != {"retrospective", "answer_only"}:
            raise RuntimeError("ordinal-binding access contrast changed")
        if report["eligible_selected_actions"] != ["A", "B", "C"]:
            raise RuntimeError("ordinal-binding actions changed")
        if report["primary_tests"] != [
            "action_label_to_response_label_lure",
            "action_position_to_response_position_lure",
        ]:
            raise RuntimeError("ordinal-binding primary family changed")
        if report["test"] != "two_sided_exact_cluster_sign_flip":
            raise RuntimeError("ordinal-binding test changed")
    else:
        raise RuntimeError("unknown self-report protocol version")
    behavior = config["behavior"]
    if behavior["prompt_variant"] != "verbose_short_cot":
        raise RuntimeError("behavior prompt variant changed")
    if int(behavior["max_new_tokens"]) > 160:
        raise RuntimeError("behavior generation budget exceeds 160 tokens")


def _validate_calibration(
    calibration: dict[str, Any], dataset: dict[str, Any]
) -> None:
    if calibration["source_dataset_sha256"] != dataset["content_sha256"]:
        raise RuntimeError("calibration does not match frozen source dataset")
    if calibration["splits"] != ["discovery"]:
        raise RuntimeError("calibration may use only the discovery split")
    if calibration["variants"] != ["verbose_short_cot"]:
        raise RuntimeError("calibration prompt variants changed")
    if int(calibration["max_new_tokens"]) > 160:
        raise RuntimeError("calibration generation budget exceeds 160 tokens")


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


def _marker_token_position(
    rendered: str, offsets: list[tuple[int, int]], marker: str
) -> int:
    marker_start = rendered.rfind(marker)
    if marker_start < 0:
        raise ValueError(f"marker text not found: {marker!r}")
    final_character = marker_start + len(marker) - 1
    positions = [
        index
        for index, (start, end) in enumerate(offsets)
        if start <= final_character < end
    ]
    if len(positions) != 1:
        raise ValueError(
            f"marker {marker!r} maps to {len(positions)} tokens at character "
            f"{final_character}"
        )
    return positions[0]


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
        encoding = tokenizer(
            rendered,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        token_ids = list(map(int, encoding["input_ids"]))
        offsets = [tuple(map(int, pair)) for pair in encoding["offset_mapping"]]
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
            name: _marker_token_position(rendered, offsets, text)
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


@app.function(
    image=image,
    volumes={"/cache": cache},
    cpu=2.0,
    memory=8192,
    timeout=600,
    max_containers=1,
    retries=0,
)
def preflight_remote(dataset: dict[str, Any], config: dict[str, Any]) -> str:
    import types

    import transformers

    _validate_config(config)
    tokenized = _tokenize_dataset(dataset, config)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION
    )
    decision_inputs = _decision_inputs(tokenizer, tokenized["rows"], config)
    dummy_behavior = [
        {
            **row,
            "parseable": True,
            "selected_action": row["expected_action"],
            "correct": True,
            "generated_text": (
                "I compared the three expected payoffs.\n\n"
                f"FINAL: {row['expected_action']}"
            ),
            "generated_tokens": 16,
            "hit_token_ceiling": False,
        }
        for row in decision_inputs
    ]
    report_inputs = _report_inputs(
        types.SimpleNamespace(tokenizer=tokenizer),
        tokenized["rows"],
        dummy_behavior,
        config,
    )
    cache.commit()
    return json.dumps(
        {
            "metadata": {
                "schema_version": 1,
                "created_at": datetime.now(UTC).isoformat(),
                "source_dataset_sha256": dataset["content_sha256"],
                "tokenized_dataset_sha256": tokenized["content_sha256"],
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
            },
            "base_prompts": {
                "n": len(decision_inputs),
                "minimum_tokens": min(
                    row["sequence_length"] for row in decision_inputs
                ),
                "maximum_tokens": max(
                    row["sequence_length"] for row in decision_inputs
                ),
            },
            "report_prompts": {
                "n": len(report_inputs),
                "minimum_tokens": min(row["sequence_length"] for row in report_inputs),
                "maximum_tokens": max(row["sequence_length"] for row in report_inputs),
            },
            "status": "all_prompt_and_continuation_checks_passed",
        },
        allow_nan=False,
        sort_keys=True,
    )


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


def _decision_inputs(
    tokenizer: Any, rows: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        task, marker_text = _verbose_task(row, short_cot=True)
        rendered = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": DECISION_SYSTEM_PROMPT},
                {"role": "user", "content": task},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        encoding = tokenizer(
            rendered,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        token_ids = list(map(int, encoding["input_ids"]))
        offsets = [tuple(map(int, pair)) for pair in encoding["offset_mapping"]]
        output.append(
            {
                **row,
                "decision_system_prompt": DECISION_SYSTEM_PROMPT,
                "decision_prompt": task,
                "prompt_token_ids": token_ids,
                "candidate_token_ids": [
                    _continuation_id(tokenizer, rendered, label) for label in "ABC"
                ],
                "marker_positions": {
                    name: _marker_token_position(rendered, offsets, text)
                    for name, text in marker_text.items()
                },
                "sequence_length": len(token_ids),
            }
        )
    return output


def _behavior_rows(
    model: Any, rows: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    import re

    import torch

    output: list[dict[str, Any]] = []
    ordered = sorted(rows, key=lambda row: (row["sequence_length"], row["condition_id"]))
    batch_size = int(config["behavior"]["batch_size"])
    maximum = int(config["behavior"]["max_new_tokens"])
    for start in range(0, len(ordered), batch_size):
        part = ordered[start : start + batch_size]
        input_ids, attention, width = _left_padded_batch(part)
        with torch.inference_mode():
            sequences = model._hf_model.generate(
                input_ids=input_ids,
                attention_mask=attention,
                do_sample=False,
                max_new_tokens=maximum,
                pad_token_id=model.tokenizer.pad_token_id,
                eos_token_id=model.tokenizer.eos_token_id,
                use_cache=True,
            )
        for index, row in enumerate(part):
            generated_ids = _trim_generated(
                [int(token) for token in sequences[index, width:].detach().cpu()],
                model.tokenizer,
            )
            generated_text = model.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )
            matches = re.findall(r"FINAL\s*:\s*([ABC])\b", generated_text, re.I)
            selected = matches[-1].upper() if matches else None
            output.append(
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"sequence_length"}
                }
                | {
                    "generated_token_ids": generated_ids,
                    "generated_text": generated_text,
                    "generated_tokens": len(generated_ids),
                    "hit_token_ceiling": len(generated_ids) == maximum,
                    "parseable": selected is not None,
                    "formatting_compliant": selected is not None,
                    "selected_action": selected,
                    "correct": selected == row["expected_action"],
                }
            )
        del input_ids, attention, sequences
    return sorted(output, key=lambda row: row["condition_id"])


def _ordinalize_trajectory(text: str) -> str:
    import re

    match = re.search(r"FINAL\s*:\s*[ABC]\b", text, re.I)
    if match is None:
        raise ValueError("cannot ordinalize an unparseable trajectory")
    body = text[: match.start()]
    final = text[match.start() :]
    names = ("first signal", "second signal", "third signal")
    for label, name in zip("ABC", names, strict=True):
        body = re.sub(rf"\bSignal\s+{label}\b", name, body, flags=re.I)
        body = re.sub(rf"\b{label}\b", name, body)
    return body + final


def _report_inputs(
    model: Any,
    dataset_rows: list[dict[str, Any]],
    behavior_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    import re

    from jspace_policy.strategic_epistemic_search import report_spec

    behavior_by_id = {row["condition_id"]: row for row in behavior_rows}
    report_config = config["self_report"]
    report_labels = tuple(map(str, report_config["candidate_labels"]))
    eligible_actions = set(report_config.get("eligible_selected_actions", "ABC"))
    naming_conditions = report_config.get("response_naming_conditions", ["indexed"])
    output = []
    for row in dataset_rows:
        if row["split"] not in report_config["splits"]:
            continue
        decision = behavior_by_id[row["condition_id"]]
        if not decision["parseable"]:
            continue
        selected_action = str(decision["selected_action"])
        if selected_action not in eligible_actions:
            continue
        matched = None
        if "matched_trajectory" in report_config["access_conditions"]:
            matched_candidates = [
                candidate
                for candidate in behavior_rows
                if candidate["condition_id"] != decision["condition_id"]
                and candidate["pair_id"] != decision["pair_id"]
                and candidate["split"] == decision["split"]
                and candidate["frame"] == decision["frame"]
                and candidate["parseable"]
                and candidate["selected_action"] == selected_action
            ]
            if not matched_candidates:
                raise RuntimeError(
                    f"no same-action matched trajectory for {decision['condition_id']}"
                )
            matched = min(
                matched_candidates,
                key=lambda candidate: (
                    abs(candidate["generated_tokens"] - decision["generated_tokens"]),
                    candidate["condition_id"],
                ),
            )
        for query_type in report_config["query_types"]:
            for access_condition in report_config["access_conditions"]:
                for response_naming in naming_conditions:
                    aliases = (
                        tuple(map(str, report_config["response_aliases"]))
                        if response_naming == "arbitrary_alias"
                        else None
                    )
                    spec = report_spec(
                        row,
                        query_type,
                        access_condition,
                        selected_action,
                        decision_prompt=decision["decision_prompt"],
                        trajectory=decision["generated_text"],
                        matched_trajectory=(
                            matched["generated_text"] if matched is not None else None
                        ),
                        ordinal_trajectory=(
                            _ordinalize_trajectory(decision["generated_text"])
                            if "ordinal_trajectory"
                            in report_config["access_conditions"]
                            else None
                        ),
                        system_prompt=decision["decision_system_prompt"],
                        report_labels=report_labels,
                        response_aliases=aliases,
                    )
                    spec["report_id"] = f"{spec['report_id']}:{response_naming}"
                    spec["response_naming"] = response_naming
                    surface_pattern = (
                        rf"(?<![A-Za-z0-9]){re.escape(spec['correct_surface'])}"
                        r"(?![A-Za-z0-9])"
                    )
                    target_surface_in_trajectory = bool(
                        re.search(
                            surface_pattern,
                            decision["generated_text"],
                            flags=re.IGNORECASE,
                        )
                    )
                    rendered = model.tokenizer.apply_chat_template(
                        spec["messages"],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    token_ids = list(
                        map(
                            int,
                            model.tokenizer.encode(rendered, add_special_tokens=False),
                        )
                    )
                    output.append(
                        {
                            **{
                                key: value
                                for key, value in spec.items()
                                if key != "messages"
                            },
                            "pair_id": row["pair_id"],
                            "pair_type": row["pair_type"],
                            "side": row["side"],
                            "split": row["split"],
                            "frame": row["frame"],
                            "decision_correct": bool(decision["correct"]),
                            "decision_generated_tokens": decision["generated_tokens"],
                            "decision_hit_token_ceiling": decision["hit_token_ceiling"],
                            "matched_control_condition_id": (
                                matched["condition_id"] if matched is not None else None
                            ),
                            "matched_control_generated_tokens": (
                                matched["generated_tokens"] if matched is not None else None
                            ),
                            "target_surface_in_trajectory": target_surface_in_trajectory,
                            "base_condition_id": row.get("base_condition_id"),
                            "base_game_id": row.get("base_game_id"),
                            "ordinal_binding": row.get("ordinal_binding"),
                            "prompt_token_ids": token_ids,
                            "candidate_token_ids": [
                                _continuation_id(model.tokenizer, rendered, label)
                                for label in report_labels
                            ],
                            "candidate_labels": list(report_labels),
                            "sequence_length": len(token_ids),
                        }
                    )
    return output


def _self_report_rows(
    model: Any,
    dataset_rows: list[dict[str, Any]],
    behavior_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    import torch

    inputs = _report_inputs(model, dataset_rows, behavior_rows, config)
    output = []
    ordered = sorted(inputs, key=lambda row: (row["sequence_length"], row["report_id"]))
    batch_size = int(config["self_report"]["batch_size"])
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
            labels = list(map(str, row["candidate_labels"]))
            candidate_logits = final[candidates]
            legal_choice_index = int(candidate_logits.argmax().cpu())
            top1 = int(final.argmax().cpu())
            top1_label = (
                labels[candidates.index(top1)] if top1 in candidates else None
            )
            expected = str(row["expected_label"])
            output.append(
                {
                    key: value
                    for key, value in row.items()
                    if key
                    not in {
                        "prompt_token_ids",
                        "candidate_token_ids",
                        "sequence_length",
                    }
                }
                | {
                    "top1_token_id": top1,
                    "top1_text": model.tokenizer.decode([top1]),
                    "formatting_compliant": top1 in candidates,
                    "top1_label": top1_label,
                    "correct": top1_label == expected,
                    "legal_choice": labels[legal_choice_index],
                    "legal_choice_correct": labels[legal_choice_index] == expected,
                    "legal_action_logits": {
                        label: float(value)
                        for label, value in zip(
                            labels, candidate_logits.detach().cpu(), strict=True
                        )
                    },
                }
            )
        del input_ids, attention, logits
    return sorted(output, key=lambda row: row["report_id"])


def _self_report_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if row["decision_correct"]]
    namings = sorted({row.get("response_naming", "indexed") for row in rows})

    def cell_key(query_type: str, access_condition: str, naming: str) -> str:
        base = f"{query_type}:{access_condition}"
        return f"{base}:{naming}" if len(namings) > 1 else base

    cells = {}
    for naming in namings:
        for query_type in sorted({row["query_type"] for row in rows}):
            for access_condition in sorted({row["access_condition"] for row in rows}):
                cell = [
                    row
                    for row in eligible
                    if row["query_type"] == query_type
                    and row["access_condition"] == access_condition
                    and row.get("response_naming", "indexed") == naming
                ]
                cells[cell_key(query_type, access_condition, naming)] = {
                    "n": len(cell),
                    "strict_accuracy": (
                        sum(row["correct"] for row in cell) / len(cell)
                        if cell
                        else None
                    ),
                    "legal_choice_accuracy": sum(
                        row["legal_choice_correct"] for row in cell
                    )
                    / len(cell)
                    if cell
                    else None,
                    "formatting_compliance": sum(
                        row["formatting_compliant"] for row in cell
                    )
                    / len(cell)
                    if cell
                    else None,
                }
    accesses = sorted({row["access_condition"] for row in rows})
    controls = [access for access in accesses if access != "retrospective"]
    contrasts = {}
    for naming in namings:
        for query_type in sorted({row["query_type"] for row in rows}):
            retrospective = cells[cell_key(query_type, "retrospective", naming)][
                "legal_choice_accuracy"
            ]
            for control in controls:
                control_accuracy = cells[cell_key(query_type, control, naming)][
                    "legal_choice_accuracy"
                ]
                key = f"{query_type}:retrospective_minus_{control}"
                if len(namings) > 1:
                    key = f"{key}:{naming}"
                contrasts[key] = (
                    retrospective - control_accuracy
                    if retrospective is not None and control_accuracy is not None
                    else None
                )
    copyability = {}
    for query_type in sorted({row["query_type"] for row in rows}):
        retrospective_rows = [
            row
            for row in eligible
            if row["query_type"] == query_type
            and row["access_condition"] == "retrospective"
        ]
        copyability[query_type] = {
            str(present).lower(): {
                "n": len(selected := [
                    row
                    for row in retrospective_rows
                    if row["target_surface_in_trajectory"] is present
                ]),
                "legal_choice_accuracy": (
                    sum(row["legal_choice_correct"] for row in selected) / len(selected)
                    if selected
                    else None
                ),
            }
            for present in (False, True)
        }
    return {
        "n_rows": len(rows),
        "n_decision_correct": len({row["condition_id"] for row in eligible}),
        "cell_accuracy": cells,
        "legal_choice_primary_contrasts": contrasts,
        "retrospective_copyability_strata": copyability,
        "interpretation": (
            "A trajectory effect is interpretable only if it survives disjoint report "
            "labels and the frozen within-decision access contrast."
        ),
    }


def _verbose_task(row: dict[str, Any], *, short_cot: bool) -> tuple[str, dict[str, str]]:
    strategic = row["frame"] == "strategic"
    actor = "receiver" if strategic else "stochastic mechanism"
    policy_heading = (
        "Conditional receiver response probabilities"
        if strategic
        else "Conditional mechanism output probabilities"
    )
    certificate = row.get("ordinal_binding") or {}
    action_order = list(map(int, certificate.get("action_presentation_order", [0, 1, 2])))
    response_order = list(
        map(int, certificate.get("response_presentation_order", [0, 1, 2]))
    )
    policy_lines = []
    for action_index in action_order:
        probabilities = row["policy"][action_index]
        rendered = ", ".join(
            f"R{index + 1}={float(Fraction(probabilities[index])):.2f}"
            for index in response_order
        )
        policy_lines.append(f"- Signal {'ABC'[action_index]}: {rendered}")
    payoff_rows = [
        f"- R{index + 1}: {int(row['payoffs'][index]):+d}"
        for index in response_order
    ]
    cost_rows = [
        f"- {'ABC'[index]}: {int(row['costs'][index])}" for index in action_order
    ]
    instruction = (
        "Reason through this in at most three short sentences and no more than "
        "40 words. Then end with exactly"
        if short_cot
        else "Do not show your work. End with exactly"
    )
    final_marker = "is A, B, or C."
    task = (
        f"Choose a signal for a {actor}.\n\n"
        f"Your payoff for each response/output is:\n" + "\n".join(payoff_rows) + "\n\n"
        f"{policy_heading}:\n" + "\n".join(policy_lines) + "\n\n"
        "Signal costs:\n" + "\n".join(cost_rows) + "\n\n"
        "For each signal, calculate the probability-weighted response payoff and "
        "subtract its cost. Select the signal with the greatest expected payoff. "
        f"{instruction} `FINAL: <letter>`, where the letter {final_marker}"
    )
    return task, {
        "payoff_end": payoff_rows[-1],
        "policy_end": policy_lines[-1],
        "final_prompt": final_marker,
    }


def _verbose_calibration_task(row: dict[str, Any]) -> str:
    return _verbose_task(row, short_cot=False)[0]


def _calibration_inputs(
    tokenizer: Any,
    dataset_rows: list[dict[str, Any]],
    calibration: dict[str, Any],
) -> list[dict[str, Any]]:
    system = DECISION_SYSTEM_PROMPT
    output = []
    for row in dataset_rows:
        if row["split"] not in calibration["splits"]:
            continue
        task_text = str(row.get("task_text", row["prompt"]))
        concise_body = task_text.rsplit("\nFor each option", 1)[0]
        prompts = {
            "concise_generated": (
                concise_body
                + "\nCompute every expected payoff carefully and check the comparison. "
                "Do not show your work. End with exactly `FINAL: <letter>`, where the "
                "letter is A, B, or C."
            ),
            "verbose_generated": _verbose_calibration_task(row),
            "verbose_short_cot": _verbose_task(row, short_cot=True)[0],
        }
        for variant in calibration["variants"]:
            rendered = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompts[variant]},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            token_ids = list(
                map(int, tokenizer.encode(rendered, add_special_tokens=False))
            )
            output.append(
                {
                    "calibration_id": f"{row['condition_id']}:{variant}",
                    "condition_id": row["condition_id"],
                    "pair_id": row["pair_id"],
                    "pair_type": row["pair_type"],
                    "side": row["side"],
                    "split": row["split"],
                    "frame": row["frame"],
                    "variant": variant,
                    "expected_action": row["expected_action"],
                    "prompt_token_ids": token_ids,
                    "sequence_length": len(token_ids),
                }
            )
    return output


def _trim_generated(token_ids: list[int], tokenizer: Any) -> list[int]:
    specials = {tokenizer.pad_token_id, tokenizer.eos_token_id}
    while token_ids and token_ids[-1] in specials:
        token_ids.pop()
    return token_ids


def _calibration_rows(
    model: Any, rows: list[dict[str, Any]], calibration: dict[str, Any]
) -> list[dict[str, Any]]:
    import re

    import torch

    output = []
    ordered = sorted(rows, key=lambda row: (row["sequence_length"], row["calibration_id"]))
    batch_size = int(calibration["batch_size"])
    maximum = int(calibration["max_new_tokens"])
    for start in range(0, len(ordered), batch_size):
        part = ordered[start : start + batch_size]
        input_ids, attention, width = _left_padded_batch(part)
        with torch.inference_mode():
            sequences = model._hf_model.generate(
                input_ids=input_ids,
                attention_mask=attention,
                do_sample=False,
                max_new_tokens=maximum,
                pad_token_id=model.tokenizer.pad_token_id,
                eos_token_id=model.tokenizer.eos_token_id,
                use_cache=True,
            )
        for index, row in enumerate(part):
            generated_ids = _trim_generated(
                [int(token) for token in sequences[index, width:].detach().cpu()],
                model.tokenizer,
            )
            generated_text = model.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )
            matches = re.findall(r"FINAL\s*:\s*([ABC])\b", generated_text, re.I)
            parsed = matches[-1].upper() if matches else None
            output.append(
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"prompt_token_ids", "sequence_length"}
                }
                | {
                    "generated_token_ids": generated_ids,
                    "generated_text": generated_text,
                    "generated_tokens": len(generated_ids),
                    "parseable": parsed is not None,
                    "selected_action": parsed,
                    "correct": parsed == row["expected_action"],
                }
            )
        del input_ids, attention, sequences
    return sorted(output, key=lambda row: row["calibration_id"])


def _calibration_summary(
    rows: list[dict[str, Any]], calibration: dict[str, Any]
) -> dict[str, Any]:
    variants = {}
    for variant in calibration["variants"]:
        selected = [row for row in rows if row["variant"] == variant]
        frames = {
            frame: sum(row["correct"] for row in selected if row["frame"] == frame)
            / sum(row["frame"] == frame for row in selected)
            for frame in sorted({row["frame"] for row in selected})
        }
        accuracy = sum(row["correct"] for row in selected) / len(selected)
        variants[variant] = {
            "n": len(selected),
            "parseable": sum(row["parseable"] for row in selected) / len(selected),
            "accuracy": accuracy,
            "frame_accuracy": frames,
            "gate_pass": sum(row["parseable"] for row in selected) / len(selected)
            >= float(calibration["minimum_parseability"])
            and accuracy >= float(calibration["minimum_variant_accuracy"])
            and min(frames.values())
            >= float(calibration["minimum_frame_accuracy"]),
        }
    passing = [name for name, value in variants.items() if value["gate_pass"]]
    selected_variant = (
        max(passing, key=lambda name: (variants[name]["accuracy"], name))
        if passing
        else None
    )
    return {
        "n_rows": len(rows),
        "variants": variants,
        "selected_variant": selected_variant,
        "gate_pass": selected_variant is not None,
    }


@app.function(
    image=image,
    volumes={"/cache": cache},
    cpu=2.0,
    memory=8192,
    timeout=600,
    max_containers=1,
    retries=0,
)
def calibration_preflight_remote(
    dataset: dict[str, Any], config: dict[str, Any], calibration: dict[str, Any]
) -> str:
    import transformers

    _validate_config(config)
    _validate_calibration(calibration, dataset)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION
    )
    rows = _calibration_inputs(tokenizer, dataset["rows"], calibration)
    cache.commit()
    return json.dumps(
        {
            "status": "all_calibration_prompts_passed",
            "n": len(rows),
            "minimum_tokens": min(row["sequence_length"] for row in rows),
            "maximum_tokens": max(row["sequence_length"] for row in rows),
            "source_dataset_sha256": dataset["content_sha256"],
            "calibration_config_sha256": _canonical_sha256(calibration),
        },
        allow_nan=False,
        sort_keys=True,
    )


@app.function(
    image=image,
    volumes={"/cache": cache},
    cpu=8.0,
    memory=32768,
    gpu="A100-80GB",
    timeout=1800,
    max_containers=1,
    retries=0,
)
def calibration_remote(
    dataset: dict[str, Any],
    config: dict[str, Any],
    calibration: dict[str, Any],
    code_metadata: dict[str, str],
) -> str:
    import torch

    _validate_config(config)
    _validate_calibration(calibration, dataset)
    started = time.perf_counter()
    model, _, metadata = _load_model(with_lens=False)
    inputs = _calibration_inputs(model.tokenizer, dataset["rows"], calibration)
    rows = _calibration_rows(model, inputs, calibration)
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
                "config_sha256": _canonical_sha256(config),
                "calibration_config_sha256": _canonical_sha256(calibration),
                "elapsed_seconds": elapsed,
                **code_metadata,
                **metadata,
            },
            "summary": _calibration_summary(rows, calibration),
            "rows": rows,
        },
        allow_nan=False,
        sort_keys=True,
    )


def _behavior_summary(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    parseable = sum(row["parseable"] for row in rows) / len(rows)
    overall = sum(row["correct"] for row in rows) / len(rows)
    parsed_rows = [row for row in rows if row["parseable"]]
    conditional = (
        sum(row["correct"] for row in parsed_rows) / len(parsed_rows)
        if parsed_rows
        else None
    )
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
        parseable >= float(config["behavior"]["minimum_parseability"])
        and overall >= float(config["behavior"]["minimum_overall_accuracy"])
        and minimum_cell >= float(config["behavior"]["minimum_cell_accuracy"])
    )
    return {
        "n_rows": len(rows),
        "parseability": parseable,
        "overall_accuracy": overall,
        "accuracy_given_parseable": conditional,
        "token_ceiling_hits": sum(row["hit_token_ceiling"] for row in rows),
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
    decision_inputs = _decision_inputs(model.tokenizer, tokenized["rows"], config)
    decision_inputs_sha256 = _canonical_sha256(
        [
            {
                "condition_id": row["condition_id"],
                "prompt_token_ids": row["prompt_token_ids"],
                "marker_positions": row["marker_positions"],
            }
            for row in decision_inputs
        ]
    )
    rows = _behavior_rows(model, decision_inputs, config)
    report_rows = _self_report_rows(model, dataset["rows"], rows, config)
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
                "decision_inputs_sha256": decision_inputs_sha256,
                "config_sha256": _canonical_sha256(config),
                "elapsed_seconds": elapsed,
                **code_metadata,
                **metadata,
            },
            "summary": {
                **_behavior_summary(rows, config),
                "self_report": _self_report_summary(report_rows),
            },
            "rows": rows,
            "self_report_rows": report_rows,
        },
        allow_nan=False,
        sort_keys=True,
    )


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
def report_remote(
    dataset: dict[str, Any],
    behavior_result: dict[str, Any],
    config: dict[str, Any],
    code_metadata: dict[str, str],
) -> str:
    """Run only the disjoint-label report fork over frozen behavior trajectories."""
    import torch

    _validate_config(config)
    if not _hash_valid(dataset):
        raise RuntimeError("source dataset hash mismatch")
    if not behavior_result["summary"]["gate_pass"]:
        raise RuntimeError("behavior gate failed; report run refused")
    if behavior_result["metadata"]["source_dataset_sha256"] != dataset["content_sha256"]:
        raise RuntimeError("behavior result does not match source dataset")
    expected_behavior_run = config["self_report"].get("source_behavior_run_id")
    if (
        expected_behavior_run is not None
        and behavior_result["metadata"]["run_id"] != expected_behavior_run
    ):
        raise RuntimeError("confirmation behavior source changed")
    started = time.perf_counter()
    model, _, metadata = _load_model(with_lens=False)
    rows = _self_report_rows(
        model,
        dataset["rows"],
        behavior_result["rows"],
        config,
    )
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
                "behavior_run_id": behavior_result["metadata"]["run_id"],
                "config_sha256": _canonical_sha256(config),
                "elapsed_seconds": elapsed,
                **code_metadata,
                **metadata,
            },
            "summary": _self_report_summary(rows),
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


def _generated_synchronization(
    generated_ids: list[int], tokenizer: Any
) -> dict[str, int]:
    import re

    first_sentence = None
    pre_final = None
    final_answer = None
    for end in range(1, len(generated_ids) + 1):
        prefix = tokenizer.decode(generated_ids[:end], skip_special_tokens=True)
        if first_sentence is None and re.search(r"[.!?](?:\s|$)", prefix):
            first_sentence = end - 1
        if pre_final is None and re.search(r"FINAL\s*:\s*$", prefix, re.I):
            pre_final = end - 1
        if final_answer is None and re.search(r"FINAL\s*:\s*[ABC]\b", prefix, re.I):
            final_answer = end - 1
    if pre_final is None or final_answer is None:
        raise RuntimeError("parseable behavior lacks FINAL synchronization tokens")
    if first_sentence is None or first_sentence >= pre_final:
        first_sentence = pre_final
    return {
        "first_reasoning_sentence": first_sentence,
        "pre_final": pre_final,
        "final_answer": final_answer,
    }


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
    behavior_rows = behavior_result["rows"]
    decision_inputs_sha256 = _canonical_sha256(
        [
            {
                "condition_id": row["condition_id"],
                "prompt_token_ids": row["prompt_token_ids"],
                "marker_positions": row["marker_positions"],
            }
            for row in behavior_rows
        ]
    )
    if decision_inputs_sha256 != behavior_result["metadata"]["decision_inputs_sha256"]:
        raise RuntimeError("decision inputs changed after behavior")
    model, lens, metadata = _load_model(with_lens=True)
    layers = list(map(int, config["mechanistic"]["layers"]))
    markers = list(config["mechanistic"]["marker_names"])
    top_k = int(config["mechanistic"]["j_lens_top_k"])
    run_id = uuid.uuid4().hex
    remote_root = Path(f"/artifacts/runs/{run_id}")
    residual_root = remote_root / "residuals"
    residual_root.mkdir(parents=True, exist_ok=True)
    output_rows = []
    ordered = sorted(
        [row for row in behavior_rows if row["parseable"]],
        key=lambda row: (
            len(row["prompt_token_ids"]) + len(row["generated_token_ids"]),
            row["condition_id"],
        ),
    )
    batch_size = int(config["behavior"]["batch_size"])
    for start in range(0, len(ordered), batch_size):
        part = ordered[start : start + batch_size]
        sequence_rows = [
            {
                "prompt_token_ids": [
                    *map(int, row["prompt_token_ids"]),
                    *map(int, row["generated_token_ids"]),
                ]
            }
            for row in part
        ]
        input_ids, attention, width = _left_padded_batch(sequence_rows)
        with torch.inference_mode(), ActivationRecorder(model.layers, at=layers) as recorder:
            outputs = model._hf_model(input_ids, attention_mask=attention)
        for batch_index, row in enumerate(part):
            prompt_ids = list(map(int, row["prompt_token_ids"]))
            generated_ids = list(map(int, row["generated_token_ids"]))
            sequence_length = len(prompt_ids) + len(generated_ids)
            offset = width - sequence_length
            generated_sync = _generated_synchronization(generated_ids, model.tokenizer)
            synchronization = {
                "policy_end": int(row["marker_positions"]["policy_end"]),
                "payoff_end": int(row["marker_positions"]["payoff_end"]),
                "final_prompt": len(prompt_ids) - 1,
                **{
                    name: len(prompt_ids) + index
                    for name, index in generated_sync.items()
                },
            }
            positions = [offset + synchronization[name] for name in markers]
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
            candidate_ids = list(map(int, row["candidate_token_ids"]))
            trajectory_positions = [
                offset + len(prompt_ids) - 1,
                *range(offset + len(prompt_ids), offset + sequence_length),
            ]
            trajectory_logits = outputs.logits[batch_index, trajectory_positions].float()[
                :, candidate_ids
            ]
            trajectory_log_probs = trajectory_logits.log_softmax(-1).detach().cpu()
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
                    "generated_text": row["generated_text"],
                    "selected_action": row["selected_action"],
                    "correct": row["correct"],
                    "synchronization": synchronization,
                    "trajectory_action_log_probs": [
                        {
                            "trace_index": trace_index,
                            "kind": "final_prompt" if trace_index == 0 else "generated_token",
                            "generated_index": trace_index - 1 if trace_index else None,
                            "surface_token": (
                                model.tokenizer.decode([generated_ids[trace_index - 1]])
                                if trace_index
                                else None
                            ),
                            "legal_action_log_probs": dict(
                                zip("ABC", map(float, values), strict=True)
                            ),
                        }
                        for trace_index, values in enumerate(trajectory_log_probs)
                    ],
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
            "decision_inputs_sha256": decision_inputs_sha256,
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
def preflight() -> None:
    config = _load_json(CONFIG_PATH)
    dataset = _load_json(DATASET_PATH)
    _validate_config(config)
    payload = json.loads(preflight_remote.remote(dataset, config))
    target = RESULT_ROOT / "raw/preflight_v2.json"
    _write_new(target, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


@app.local_entrypoint()
def calibration_preflight() -> None:
    config = _load_json(CONFIG_PATH)
    calibration_config = _load_json(CALIBRATION_CONFIG_PATH)
    dataset = _load_json(DATASET_PATH)
    _validate_config(config)
    _validate_calibration(calibration_config, dataset)
    payload = json.loads(
        calibration_preflight_remote.remote(dataset, config, calibration_config)
    )
    target = RESULT_ROOT / "raw/calibration_v3_preflight.json"
    _write_new(target, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


@app.local_entrypoint()
def calibration() -> None:
    config = _load_json(CONFIG_PATH)
    calibration_config = _load_json(CALIBRATION_CONFIG_PATH)
    dataset = _load_json(DATASET_PATH)
    _validate_config(config)
    _validate_calibration(calibration_config, dataset)
    ledger = _admit(config, "calibration", 1800, 32)
    code_metadata = {"git_commit": _git_head(), "worktree_sha256": _worktree_sha256()}
    payload = json.loads(
        calibration_remote.remote(
            dataset, config, calibration_config, code_metadata
        )
    )
    target = RESULT_ROOT / "raw/calibration_v3.json"
    _write_new(target, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _record_cost(ledger, config, payload, "calibration", 32)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


@app.local_entrypoint()
def behavior() -> None:
    config = _load_json(CONFIG_PATH)
    dataset = _load_json(DATASET_PATH)
    _validate_config(config)
    ledger = _admit(config, "behavior", 1800, 32)
    code_metadata = {"git_commit": _git_head(), "worktree_sha256": _worktree_sha256()}
    payload = json.loads(behavior_remote.remote(dataset, config, code_metadata))
    target = RESULT_ROOT / "raw/behavior_v2.json"
    _write_new(target, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _record_cost(ledger, config, payload, "behavior", 32)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


def _confirmation_config() -> dict[str, Any]:
    config = _load_json(CONFIG_PATH)
    confirmation = _load_json(REPORT_CONFIRMATION_CONFIG_PATH)
    config["self_report"] = confirmation["self_report"]
    return config


def _alias_mechanism_config() -> dict[str, Any]:
    config = _load_json(CONFIG_PATH)
    protocol = _load_json(REPORT_ALIAS_CONFIG_PATH)
    config["self_report"] = protocol["self_report"]
    return config


def _ordinal_binding_config() -> dict[str, Any]:
    return _load_json(ORDINAL_CONFIG_PATH)


@app.local_entrypoint()
def report_preflight() -> None:
    config = _load_json(CONFIG_PATH)
    dataset = _load_json(DATASET_PATH)
    _validate_config(config)
    payload = json.loads(preflight_remote.remote(dataset, config))
    target = RESULT_ROOT / "raw/report_disjoint_v1_preflight.json"
    _write_new(target, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


@app.local_entrypoint()
def report() -> None:
    config = _load_json(CONFIG_PATH)
    dataset = _load_json(DATASET_PATH)
    behavior_result = _load_json(RESULT_ROOT / "raw/behavior_v2.json")
    _validate_config(config)
    ledger = _admit(config, "report_disjoint_v1", 1800, 32)
    code_metadata = {"git_commit": _git_head(), "worktree_sha256": _worktree_sha256()}
    payload = json.loads(
        report_remote.remote(dataset, behavior_result, config, code_metadata)
    )
    target = RESULT_ROOT / "raw/report_disjoint_v1.json"
    _write_new(target, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _record_cost(ledger, config, payload, "report_disjoint_v1", 32)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


@app.local_entrypoint()
def report_confirmation_preflight() -> None:
    config = _confirmation_config()
    dataset = _load_json(DATASET_PATH)
    _validate_config(config)
    payload = json.loads(preflight_remote.remote(dataset, config))
    target = RESULT_ROOT / "raw/report_confirmation_v1_preflight.json"
    _write_new(target, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


@app.local_entrypoint()
def report_confirmation() -> None:
    config = _confirmation_config()
    dataset = _load_json(DATASET_PATH)
    behavior_result = _load_json(RESULT_ROOT / "raw/behavior_v2.json")
    _validate_config(config)
    ledger = _admit(config, "report_confirmation_v1", 900, 32)
    code_metadata = {"git_commit": _git_head(), "worktree_sha256": _worktree_sha256()}
    payload = json.loads(
        report_remote.remote(dataset, behavior_result, config, code_metadata)
    )
    target = RESULT_ROOT / "raw/report_confirmation_v1.json"
    _write_new(target, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _record_cost(ledger, config, payload, "report_confirmation_v1", 32)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


@app.local_entrypoint()
def report_alias_preflight() -> None:
    config = _alias_mechanism_config()
    dataset = _load_json(DATASET_PATH)
    _validate_config(config)
    payload = json.loads(preflight_remote.remote(dataset, config))
    target = RESULT_ROOT / "raw/report_alias_mechanism_v1_preflight.json"
    _write_new(target, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


@app.local_entrypoint()
def report_alias() -> None:
    config = _alias_mechanism_config()
    dataset = _load_json(DATASET_PATH)
    behavior_result = _load_json(RESULT_ROOT / "raw/behavior_v2.json")
    _validate_config(config)
    ledger = _admit(config, "report_alias_mechanism_v1", 900, 32)
    code_metadata = {"git_commit": _git_head(), "worktree_sha256": _worktree_sha256()}
    payload = json.loads(
        report_remote.remote(dataset, behavior_result, config, code_metadata)
    )
    target = RESULT_ROOT / "raw/report_alias_mechanism_v1.json"
    _write_new(target, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _record_cost(ledger, config, payload, "report_alias_mechanism_v1", 32)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


@app.local_entrypoint()
def freeze_ordinal_dataset() -> None:
    from jspace_policy.strategic_epistemic_search import (
        dataset_payload,
        verify_dataset_payload,
    )

    config = _ordinal_binding_config()
    _validate_config(config)
    payload = dataset_payload(config)
    verify_dataset_payload(payload, config)
    _write_new(
        ORDINAL_DATASET_PATH, json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {ORDINAL_DATASET_PATH} with {len(payload['rows'])} rows")


@app.local_entrypoint()
def ordinal_preflight() -> None:
    config = _ordinal_binding_config()
    dataset = _load_json(ORDINAL_DATASET_PATH)
    _validate_config(config)
    payload = json.loads(preflight_remote.remote(dataset, config))
    target = RESULT_ROOT / "raw/ordinal_binding_v1_preflight.json"
    _write_new(target, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


@app.local_entrypoint()
def ordinal_behavior() -> None:
    config = _ordinal_binding_config()
    dataset = _load_json(ORDINAL_DATASET_PATH)
    _validate_config(config)
    ledger = _admit(config, "ordinal_binding_behavior_v1", 1800, 32)
    code_metadata = {"git_commit": _git_head(), "worktree_sha256": _worktree_sha256()}
    payload = json.loads(behavior_remote.remote(dataset, config, code_metadata))
    target = RESULT_ROOT / "raw/ordinal_binding_behavior_v1.json"
    _write_new(target, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _record_cost(ledger, config, payload, "ordinal_binding_behavior_v1", 32)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


@app.local_entrypoint()
def ordinal_report() -> None:
    config = _ordinal_binding_config()
    dataset = _load_json(ORDINAL_DATASET_PATH)
    behavior_result = _load_json(RESULT_ROOT / "raw/ordinal_binding_behavior_v1.json")
    _validate_config(config)
    ledger = _admit(config, "ordinal_binding_report_v1", 1200, 32)
    code_metadata = {"git_commit": _git_head(), "worktree_sha256": _worktree_sha256()}
    payload = json.loads(
        report_remote.remote(dataset, behavior_result, config, code_metadata)
    )
    target = RESULT_ROOT / "raw/ordinal_binding_report_v1.json"
    _write_new(target, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _record_cost(ledger, config, payload, "ordinal_binding_report_v1", 32)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


@app.local_entrypoint()
def mechanistic() -> None:
    config = _load_json(CONFIG_PATH)
    dataset = _load_json(DATASET_PATH)
    behavior_result = _load_json(RESULT_ROOT / "raw/behavior_v2.json")
    _validate_config(config)
    ledger = _admit(config, "mechanistic", 3600, 64)
    code_metadata = {"git_commit": _git_head(), "worktree_sha256": _worktree_sha256()}
    payload = json.loads(
        mechanistic_remote.remote(dataset, behavior_result, config, code_metadata)
    )
    target = RESULT_ROOT / "raw/mechanistic_v2_manifest.json"
    _write_new(target, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _record_cost(ledger, config, payload, "mechanistic", 64)
    print(json.dumps(payload, indent=2, sort_keys=True))
