"""Immutable analysis outputs for report-reactivity CPU joins."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jspace_policy.sprint_runtime import load_json_file, write_new


def test_write_new_refuses_existing_analysis_path(tmp_path: Path) -> None:
    path = tmp_path / "analysis" / "arm_accuracy_summary.json"
    write_new(path, {"first": True})
    with pytest.raises(FileExistsError):
        write_new(path, {"first": False})
    assert json.loads(path.read_text())["first"] is True


def test_harder_games_analyze_fails_closed_without_unlink(tmp_path: Path) -> None:
    """Regression: analyze_harder_games must not unlink before write_new."""
    source = Path("experiments/report_reactivity/analyze_harder_games.py").read_text()
    assert "unlink(" not in source
    assert "write_new(args.output, summary)" in source


def test_mid_trajectory_analyze_fails_closed_without_unlink(tmp_path: Path) -> None:
    """Regression: analyze_ask_mid_trajectory must not unlink before write_new."""
    source = Path(
        "experiments/report_reactivity/analyze_ask_mid_trajectory.py"
    ).read_text()
    assert "unlink(" not in source
    assert "write_new(args.output, summary)" in source


def test_committed_prepared_matches_raw_payload_hash() -> None:
    root = Path("results/report_reactivity")
    for run_id in (
        "gha-report16-38-v1",
        "harder-games-qwen38-n16-v1",
        "ask-mid-traj-qwen38-n16-v1",
    ):
        prepared = load_json_file(root / run_id / "prepared.json.gz")
        raw = json.loads((root / run_id / "raw.json").read_text())
        assert prepared["sha256"] == raw["payload_sha256"]
