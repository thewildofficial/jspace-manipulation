"""Fail-closed Actions gates for report-reactivity paid runs.

CPU prepare / dry_run stays allowed. GPU scoring requires:
1. Task-specific unlock (rename has no analyzer/results path yet).
2. A reviewed pinned prepare sha256 that matches the freshly prepared payload.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Unlock rename GPU only when analyze_rename_invariant.py + documented results
# layout exist AND instrument parity issues are understood. Do not flip this
# for a stub analyzer.
RENAME_GPU_SCORING_UNLOCKED = False
RENAME_ANALYZER_RELPATH = "experiments/report_reactivity/analyze_rename_invariant.py"


def rename_analyzer_exists(repo_root: Path | None = None) -> bool:
    root = Path(".") if repo_root is None else repo_root
    return (root / RENAME_ANALYZER_RELPATH).is_file()


def assert_rename_gpu_allowed(*, task: str, dry_run: bool) -> None:
    """Refuse paid rename_invariant scoring until the reporting path is ready."""

    if task != "rename_invariant" or dry_run:
        return
    if RENAME_GPU_SCORING_UNLOCKED and rename_analyzer_exists():
        return
    reasons: list[str] = []
    if not RENAME_GPU_SCORING_UNLOCKED:
        reasons.append("RENAME_GPU_SCORING_UNLOCKED=false")
    if not rename_analyzer_exists():
        reasons.append(f"missing {RENAME_ANALYZER_RELPATH}")
    detail = "; ".join(reasons) if reasons else "reporting path incomplete"
    raise SystemExit(
        "refusing task=rename_invariant with dry_run=false: no complete "
        f"scoring/reporting path yet ({detail}). CPU dry_run/prepare remains "
        "allowed. Unlock only after analyzer + methods/results layout exist."
    )


def prepared_payload_sha256(prepared_path: Path) -> str:
    payload: dict[str, Any] = json.loads(prepared_path.read_text())
    sha = payload.get("sha256")
    if not isinstance(sha, str) or not sha:
        raise SystemExit(f"prepared payload missing sha256: {prepared_path}")
    return sha


def assert_pinned_prepare_sha(
    *,
    dry_run: bool,
    pinned_payload_sha256: str,
    prepared_sha256: str,
) -> None:
    """When spending GPU, require an explicit reviewed prepare hash match."""

    if dry_run:
        return
    pinned = (pinned_payload_sha256 or "").strip().lower()
    actual = (prepared_sha256 or "").strip().lower()
    if not pinned:
        raise SystemExit(
            "dry_run=false requires input pinned_payload_sha256 (sha256 of a "
            "reviewed dry-run / committed prepared payload). Re-run with "
            "dry_run=true, record the prepare sha256, then pass it back."
        )
    if pinned != actual:
        raise SystemExit(
            "pinned_payload_sha256 mismatch: "
            f"pinned={pinned} prepared={actual}. Fail closed — do not score "
            "an unreviewed prepare. Re-prepare, re-review, and update the pin."
        )


def enforce_paid_run_gates(
    *,
    task: str,
    dry_run: bool,
    pinned_payload_sha256: str,
    prepared_path: Path,
) -> dict[str, str]:
    """Run all paid-execution gates; return checked hashes for logging."""

    assert_rename_gpu_allowed(task=task, dry_run=dry_run)
    prepared_sha = prepared_payload_sha256(prepared_path)
    assert_pinned_prepare_sha(
        dry_run=dry_run,
        pinned_payload_sha256=pinned_payload_sha256,
        prepared_sha256=prepared_sha,
    )
    return {
        "prepared_sha256": prepared_sha,
        "pinned_payload_sha256": pinned_payload_sha256,
    }


__all__ = [
    "RENAME_ANALYZER_RELPATH",
    "RENAME_GPU_SCORING_UNLOCKED",
    "assert_pinned_prepare_sha",
    "assert_rename_gpu_allowed",
    "enforce_paid_run_gates",
    "prepared_payload_sha256",
    "rename_analyzer_exists",
]
