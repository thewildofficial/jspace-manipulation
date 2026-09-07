"""Fail-closed provenance and reservation accounting for the next sprint."""

from __future__ import annotations

import fcntl
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Any

STAGE_LIMITS = {
    "preflight": 2.0,
    "baseline": 4.0,
    "incident": 8.0,
    "mechanistic": 6.0,
    "replication": 4.0,
    "overhead": 6.0,
}
# Historical local/manual study ledger (immutable rows; do not rewrite).
DEFAULT_LEDGER_PATH = Path("results/report_reactivity/reservations.jsonl")
DEFAULT_GLOBAL_CEILING_USD = 30.0
# GHA-era ledger: fresh counters; global ceiling tracks ~$28 Modal balance.
GHA_LEDGER_PATH = Path("results/report_reactivity/reservations_gha.jsonl")
GHA_GLOBAL_CEILING_USD = 28.0
LEDGER_ENV = "REPORT_REACTIVITY_LEDGER"
GLOBAL_CEILING_ENV = "REPORT_REACTIVITY_GLOBAL_CEILING"
MODEL_REVISIONS = {
    "Qwen/Qwen3.6-27B": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
    "Qwen/Qwen3.8-27B": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
}


def resolve_ledger_path() -> Path:
    """Ledger path from ``REPORT_REACTIVITY_LEDGER``, else the historical default."""

    raw = os.environ.get(LEDGER_ENV)
    return Path(raw) if raw else DEFAULT_LEDGER_PATH


def global_ceiling_usd_for(path: Path | None = None) -> float:
    """Global USD ceiling for a ledger file.

    Explicit ``REPORT_REACTIVITY_GLOBAL_CEILING`` wins. Otherwise the GHA ledger
    filename maps to 28.0 and the historical ledger (default) maps to 30.0.
    """

    override = os.environ.get(GLOBAL_CEILING_ENV)
    if override is not None and override != "":
        value = float(override)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("invalid global ceiling override")
        return value
    ledger = path if path is not None else resolve_ledger_path()
    if ledger.name == GHA_LEDGER_PATH.name:
        return GHA_GLOBAL_CEILING_USD
    return DEFAULT_GLOBAL_CEILING_USD


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def to_jsonable(value: object) -> object:
    """Recursively convert tensors/arrays to plain JSON types (no torch required).

    Modal's default pickle return path fails on Actions CPU runners that lack
    torch even when values look like Python floats. Prefer sanitizing here and
    returning ``dumps_jsonable(...)`` (a JSON string) from remote GPU functions.
    """

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    shape = getattr(value, "shape", None)
    tolist = getattr(value, "tolist", None)
    item = getattr(value, "item", None)
    if shape is not None and callable(tolist):
        # torch.Tensor / numpy.ndarray: scalars via .item(), else .tolist().
        if tuple(shape) == ():
            if not callable(item):
                raise TypeError(f"scalar array without .item(): {type(value).__name__}")
            return to_jsonable(item())
        return to_jsonable(tolist())
    if callable(item) and not callable(tolist):
        return to_jsonable(item())
    if callable(tolist):
        return to_jsonable(tolist())
    raise TypeError(f"non-JSON-serializable type: {type(value).__name__}")


def dumps_jsonable(value: object) -> str:
    """Serialize ``value`` as a JSON string after ``to_jsonable`` sanitization."""

    return json.dumps(to_jsonable(value), allow_nan=False, sort_keys=True)


def loads_jsonable(payload: str | dict) -> dict:
    """Parse a Modal score return (JSON string preferred; dict accepted)."""

    if isinstance(payload, str):
        loaded = json.loads(payload)
    elif isinstance(payload, dict):
        loaded = payload
    else:
        raise TypeError(f"expected JSON str or dict, got {type(payload).__name__}")
    if not isinstance(loaded, dict):
        raise TypeError("score payload must decode to a dict")
    return loaded


def write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def load_json_file(path: Path) -> Any:
    """Load ordinary JSON or a gzip-compressed JSON archive."""

    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def write_gzip_new(path: Path, value: object) -> None:
    """Write deterministic gzip JSON without replacing an existing artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as stream:
                json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")


def reserve(
    path: Path,
    run_id: str,
    stage: str,
    ceiling_usd: float,
    *,
    global_ceiling_usd: float | None = None,
) -> dict:
    """Reserve before dispatch; failures/unknown charges retain the entire ceiling.

    This ledger never releases a reservation from an elapsed-time estimate. Provider
    billing reconciliation is recorded separately and cannot silently refund it.
    flock serializes local dispatches; use a single coordinator for this study.
    """
    if stage not in STAGE_LIMITS or not math.isfinite(ceiling_usd) or ceiling_usd <= 0:
        raise ValueError("invalid reservation")
    ceiling_cap = (
        global_ceiling_usd
        if global_ceiling_usd is not None
        else global_ceiling_usd_for(path)
    )
    if not math.isfinite(ceiling_cap) or ceiling_cap <= 0:
        raise ValueError("invalid global ceiling")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        entries = [json.loads(line) for line in stream if line.strip()]
        if any(row["run_id"] == run_id for row in entries):
            raise ValueError("run ID already reserved; automatic retry refused")
        total = sum(row["ceiling_usd"] for row in entries)
        stage_total = sum(row["ceiling_usd"] for row in entries if row["stage"] == stage)
        if total + ceiling_usd > ceiling_cap or stage_total + ceiling_usd > STAGE_LIMITS[stage]:
            raise ValueError("global or stage budget exhausted")
        row = {
            "run_id": run_id,
            "stage": stage,
            "ceiling_usd": ceiling_usd,
            "status": "reserved_unreconciled",
            "total_reserved_usd": total + ceiling_usd,
            "ledger_path": str(path),
            "global_ceiling_usd": ceiling_cap,
        }
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
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
