from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

DEVIATION = (
    "The ordinal behavior entrypoint called behavior_remote(), which generated "
    "the preregistered self-report rows in the same Modal invocation. The named "
    "ordinal-report workflow step was therefore skipped, but its outcomes were "
    "already present in the immutable behavior artifact."
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract(payload: dict[str, Any], source_sha256: str) -> dict[str, Any]:
    if not payload.get("summary", {}).get("gate_pass"):
        raise ValueError("behavior gate did not pass; embedded report is not licensed")
    rows = payload.get("self_report_rows")
    summary = payload.get("summary", {}).get("self_report")
    if not isinstance(rows, list) or not rows or not isinstance(summary, dict):
        raise ValueError("source payload does not contain an embedded self-report")
    metadata = dict(payload["metadata"])
    behavior_run_id = str(metadata["run_id"])
    metadata.update(
        {
            "artifact_kind": "lossless_embedded_report_extraction",
            "behavior_run_id": behavior_run_id,
            "protocol_deviation": DEVIATION,
            "source_behavior_payload_sha256": source_sha256,
            "source_stage": "ordinal_binding_behavior_v1",
        }
    )
    return {
        "metadata": metadata,
        "summary": summary,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = extract(payload, file_sha256(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
