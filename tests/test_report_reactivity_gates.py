"""Paid-run gates: rename GPU unlock + pinned prepare sha256."""

from __future__ import annotations

from pathlib import Path

import pytest

from jspace_policy.rename_invariant import LEGACY_STUDY_ID, PROTOCOL_ID, STUDY_ID
from jspace_policy.report_reactivity import (
    MID_TRAJECTORY_LEGACY_STUDY_ID,
    MID_TRAJECTORY_STUDY_ID,
)
from jspace_policy.report_reactivity_gates import (
    RENAME_ANALYZER_RELPATH,
    RENAME_GPU_SCORING_UNLOCKED,
    assert_pinned_prepare_sha,
    assert_rename_gpu_allowed,
    enforce_paid_run_gates,
    prepared_payload_sha256,
    rename_analyzer_exists,
)


def test_rename_gpu_scoring_remains_locked() -> None:
    assert RENAME_GPU_SCORING_UNLOCKED is False
    assert not Path(RENAME_ANALYZER_RELPATH).exists()
    assert rename_analyzer_exists() is False


def test_rename_gpu_refuses_paid_run() -> None:
    with pytest.raises(SystemExit, match="refusing task=rename_invariant"):
        assert_rename_gpu_allowed(task="rename_invariant", dry_run=False)


def test_rename_gpu_allows_dry_run_and_other_tasks() -> None:
    assert_rename_gpu_allowed(task="rename_invariant", dry_run=True)
    assert_rename_gpu_allowed(task="ask_mid_trajectory", dry_run=False)
    assert_rename_gpu_allowed(task="report", dry_run=False)


def test_pinned_prepare_sha_required_when_not_dry_run() -> None:
    with pytest.raises(SystemExit, match="pinned_payload_sha256"):
        assert_pinned_prepare_sha(
            dry_run=False,
            pinned_payload_sha256="",
            prepared_sha256="abc",
        )
    with pytest.raises(SystemExit, match="mismatch"):
        assert_pinned_prepare_sha(
            dry_run=False,
            pinned_payload_sha256="deadbeef",
            prepared_sha256="abc",
        )
    assert_pinned_prepare_sha(
        dry_run=False,
        pinned_payload_sha256="AbC",
        prepared_sha256="abc",
    )
    assert_pinned_prepare_sha(
        dry_run=True,
        pinned_payload_sha256="",
        prepared_sha256="ignored",
    )


def test_enforce_paid_run_gates_reads_prepared_sha(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared.json"
    prepared.write_text('{"sha256": "feedface"}\n')
    assert prepared_payload_sha256(prepared) == "feedface"
    with pytest.raises(SystemExit, match="refusing task=rename_invariant"):
        enforce_paid_run_gates(
            task="rename_invariant",
            dry_run=False,
            pinned_payload_sha256="feedface",
            prepared_path=prepared,
        )
    checked = enforce_paid_run_gates(
        task="ask_mid_trajectory",
        dry_run=False,
        pinned_payload_sha256="feedface",
        prepared_path=prepared,
    )
    assert checked["prepared_sha256"] == "feedface"


def test_canonical_study_ids_match_protocol_ids() -> None:
    assert MID_TRAJECTORY_STUDY_ID == "ask_mid_trajectory"
    assert MID_TRAJECTORY_LEGACY_STUDY_ID == "ASK-MID-TRAJECTORY-1"
    assert STUDY_ID == PROTOCOL_ID == "rename_invariant"
    assert LEGACY_STUDY_ID == "RENAME-INVARIANT-1"


def test_naming_doc_and_protocol_json_agree() -> None:
    naming = Path("experiments/report_reactivity/NAMING.md").read_text()
    assert "`ask_mid_trajectory`" in naming
    assert "`rename_invariant`" in naming
    assert "PriGo" not in naming or "never put operator" in naming.lower()
    mid = Path(
        "experiments/report_reactivity/protocol_ask_mid_trajectory.json"
    ).read_text()
    rename = Path(
        "experiments/report_reactivity/protocol_rename_invariant.json"
    ).read_text()
    assert '"study_id": "ask_mid_trajectory"' in mid
    assert '"protocol_id": "ask_mid_trajectory"' in mid
    assert '"study_id": "rename_invariant"' in rename
    assert '"protocol_id": "rename_invariant"' in rename


def test_science_docs_omit_queue_ops_voice() -> None:
    paths = [
        "experiments/report_reactivity/ask-mid-trajectory-protocol.md",
        "experiments/report_reactivity/ask-mid-trajectory-story.md",
        "experiments/report_reactivity/rename-invariant-protocol.md",
        "experiments/report_reactivity/README.md",
        "docs/next-sprint/experiments.md",
    ]
    forbidden = ("PriGo", "#6", "#7", "RENAME-INVARIANT-1", "ASK-MID-TRAJECTORY-1")
    for path in paths:
        text = Path(path).read_text()
        for token in ("PriGo", "#6", "#7"):
            assert token not in text, f"{path} still mentions ops token {token}"
        # Legacy study labels may appear only in explicit alias tables.
        if "ASK-MID-TRAJECTORY-1" in text or "RENAME-INVARIANT-1" in text:
            assert "Legacy" in text or "legacy" in text or "Historical" in text, path
    # Claim-ledger claim rows keep C##; must not push queue slang into science.
    ledger = Path("docs/next-sprint/claim-ledger.md").read_text()
    for token in ("PriGo", "#6", "#7"):
        assert token not in ledger
    # Decision log may keep ops language, but must label it.
    decision = Path("docs/next-sprint/decision-log.md").read_text()
    assert "PriGo" in decision
    assert "operator" in decision.lower() or "ops" in decision.lower()
    _ = forbidden  # documented intent; path loop above is the assertion
