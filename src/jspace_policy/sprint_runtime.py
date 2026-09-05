"""Fail-closed provenance and reservation accounting for the next sprint."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
from pathlib import Path

STAGE_LIMITS = {
    "preflight": 2.0,
    "baseline": 4.0,
    "incident": 8.0,
    "mechanistic": 6.0,
    "replication": 4.0,
    "overhead": 6.0,
}
MODEL_REVISIONS = {
    "Qwen/Qwen3.6-27B": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
    "Qwen/Qwen3.8-27B": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
}


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def reserve(path: Path, run_id: str, stage: str, ceiling_usd: float) -> dict:
    """Reserve before dispatch; failures/unknown charges retain the entire ceiling.

    This ledger never releases a reservation from an elapsed-time estimate. Provider
    billing reconciliation is recorded separately and cannot silently refund it.
    flock serializes local dispatches; use a single coordinator for this study.
    """
    if stage not in STAGE_LIMITS or not math.isfinite(ceiling_usd) or ceiling_usd <= 0:
        raise ValueError("invalid reservation")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        entries = [json.loads(line) for line in stream if line.strip()]
        if any(row["run_id"] == run_id for row in entries):
            raise ValueError("run ID already reserved; automatic retry refused")
        total = sum(row["ceiling_usd"] for row in entries)
        stage_total = sum(row["ceiling_usd"] for row in entries if row["stage"] == stage)
        if total + ceiling_usd > 30 or stage_total + ceiling_usd > STAGE_LIMITS[stage]:
            raise ValueError("global or stage budget exhausted")
        row = {
            "run_id": run_id,
            "stage": stage,
            "ceiling_usd": ceiling_usd,
            "status": "reserved_unreconciled",
            "total_reserved_usd": total + ceiling_usd,
        }
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        import os

        os.fsync(stream.fileno())
        return row


def prepare_query(tokenizer, messages: list[dict], labels=("A", "B")) -> dict:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
        preserve_thinking=False,
    )
    tokens = tokenizer.encode(rendered, add_special_tokens=False)
    candidates = []
    for label in labels:
        full = tokenizer.encode(rendered + label, add_special_tokens=False)
        if full[:-1] != tokens:
            raise ValueError(f"candidate {label!r} is not an exact single-token continuation")
        candidates.append(full[-1])
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidate token collision")
    return {
        "input_ids": tokens,
        "candidate_ids": candidates,
        "labels": list(labels),
        "rendered_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        "length": len(tokens),
    }


def verify_payload(payload: dict) -> None:
    body = {key: value for key, value in payload.items() if key != "sha256"}
    if payload.get("sha256") != digest(body):
        raise ValueError("payload hash mismatch")
    if MODEL_REVISIONS.get(payload["model_id"]) != payload["revision"]:
        raise ValueError("unfrozen checkpoint")
    if (payload["split"], payload["status"]) == ("discovery", "engineering_pilot"):
        pass
    elif (payload["split"], payload["status"]) == ("locked", "confirmation"):
        contrast = payload.get("frozen_contrast")
        if not isinstance(contrast, dict) or not contrast.get("contrast_id"):
            raise ValueError("locked confirmation requires a frozen contrast")
    else:
        raise ValueError("runner does not authorize mechanistic inference")
    if not payload["queries"]:
        raise ValueError("empty inference payload")
