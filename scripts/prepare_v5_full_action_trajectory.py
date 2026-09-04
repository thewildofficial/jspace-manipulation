from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jspace_policy.full_action_trajectory import (
    ACTION_LABELS,
    REPORT_LABELS,
    canonical_sha256,
    expected_reports,
    first_report_messages,
    swapped_reports,
    trajectory_messages,
    verify_dataset_payload,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v5/full_action_trajectory/experiment.json"
DATASET = ROOT / "configs/v5/full_action_trajectory/dataset.json"
OUTPUT = ROOT / "artifacts/processed/v5_full_action_trajectory_prepared.json"
PREFLIGHT = ROOT / "results/v5_full_action_trajectory/raw/preflight_v1.json"
SYSTEM_PROMPT = (
    "Treat the described game and prior episodes as exact formal information. "
    "Follow the requested forced-choice format exactly."
)


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
    candidates: tuple[str, str],
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
        "candidate_labels": list(candidates),
        "candidate_token_ids": [
            _continuation_id(tokenizer, rendered, candidate) for candidate in candidates
        ],
        "sequence_length": len(token_ids),
    }


def _pair_key(reports: dict[str, str]) -> str:
    return f"A:{reports['A']}|B:{reports['B']}"


def build_prepared(
    tokenizer: Any, config: dict[str, Any], dataset: dict[str, Any]
) -> dict[str, Any]:
    prepared_rows = []
    sequence_lengths = []
    branch_count = 0
    for row in dataset["rows"]:
        trajectory_id = row["trajectory_id"]
        first_action = row["report_order"][0]
        first = _prepare_query(
            tokenizer,
            f"{trajectory_id}:report:{first_action}:first",
            first_report_messages(row, SYSTEM_PROMPT),
            REPORT_LABELS,
        )
        second_by_first = {}
        for first_choice in REPORT_LABELS:
            messages = trajectory_messages(
                row,
                SYSTEM_PROMPT,
                {first_action: first_choice},
            )
            second_action = row["report_order"][1]
            second_by_first[first_choice] = _prepare_query(
                tokenizer,
                f"{trajectory_id}:report:{second_action}:after:{first_choice}",
                messages,
                REPORT_LABELS,
            )
        action_by_reports = {}
        for a_choice in REPORT_LABELS:
            for b_choice in REPORT_LABELS:
                reports = {"A": a_choice, "B": b_choice}
                key = _pair_key(reports)
                action_by_reports[key] = _prepare_query(
                    tokenizer,
                    f"{trajectory_id}:action:{key}",
                    trajectory_messages(row, SYSTEM_PROMPT, reports),
                    ACTION_LABELS,
                )
        queries = [first, *second_by_first.values(), *action_by_reports.values()]
        sequence_lengths.extend(query["sequence_length"] for query in queries)
        branch_count += len(queries)
        prepared_rows.append(
            {
                "trajectory_id": trajectory_id,
                "first_report": first,
                "second_report_by_first_choice": second_by_first,
                "action_by_report_pair": action_by_reports,
                "oracle_pair_key": _pair_key(expected_reports(row)),
                "swapped_pair_key": _pair_key(swapped_reports(row)),
            }
        )
    body = {
        "schema_version": 1,
        "study_id": "V5-RBG-6",
        "config_sha256": canonical_sha256(config),
        "dataset_sha256": dataset["content_sha256"],
        "model_id": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "pad_token_id": int(tokenizer.pad_token_id),
        "rows": prepared_rows,
    }
    return {
        **body,
        "content_sha256": canonical_sha256(body),
        "preflight": {
            "queries_validated": branch_count,
            "minimum_tokens": min(sequence_lengths),
            "maximum_tokens": max(sequence_lengths),
        },
    }


def main() -> None:
    import transformers

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--preflight", type=Path, default=PREFLIGHT)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    verify_dataset_payload(dataset, config)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config["model"]["id"], revision=config["model"]["revision"]
    )
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    prepared = build_prepared(tokenizer, config, dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(prepared, separators=(",", ":")) + "\n")
    if args.preflight.exists():
        raise SystemExit(f"refusing to overwrite preflight: {args.preflight}")
    preflight = {
        "schema_version": 1,
        "study_id": "V5-RBG-6",
        "created_at": datetime.now(UTC).isoformat(),
        "execution_location": "github_actions_cpu",
        "source_dataset_sha256": dataset["content_sha256"],
        "prepared_payload_sha256": prepared["content_sha256"],
        "model_id": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "transformers_version": transformers.__version__,
        **prepared["preflight"],
        "status": "all_branch_prompts_and_one_token_continuations_validated_locally",
    }
    args.preflight.parent.mkdir(parents=True, exist_ok=True)
    args.preflight.write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n")
    print(json.dumps(preflight, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
