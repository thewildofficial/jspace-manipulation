from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from jspace_policy.budget import estimate_cost
from jspace_policy.mechanistic_decomposition import analyze_behavior
from jspace_policy.mechanistic_decomposition_analysis import (
    _resolve_remote_artifact,
    compute_activation_geometry,
)
from jspace_policy.mechanistic_decomposition_games import (
    RBG5B_CONCEPT_PAIRS,
    dataset_payload,
    verify_dataset_payload,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v5/mechanistic_decomposition_b/experiment.json"
MANIFEST = ROOT / "configs/v5/mechanistic_decomposition_b/dataset_manifest.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_rbg5b_dataset_is_fresh_deterministic_and_hashed() -> None:
    config = _config()
    payload = dataset_payload(config)
    verify_dataset_payload(payload, config)
    assert payload == dataset_payload(config)
    assert len(RBG5B_CONCEPT_PAIRS) == 96
    assert len(payload["rows"]) == 1344
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert payload["content_sha256"] == manifest["expected_content_sha256"]
    assert payload["config_sha256"] == manifest["expected_config_sha256"]
    discovery = {
        concept
        for row in payload["rows"]
        if row["split"] == "discovery"
        for concept in row["concepts"]
    }
    locked = {
        concept
        for row in payload["rows"]
        if row["split"] == "locked"
        for concept in row["concepts"]
    }
    assert discovery.isdisjoint(locked)


def test_rbg5b_capture_factorial_and_report_gate() -> None:
    config = _config()
    dataset = dataset_payload(config)
    rows = []
    for source in dataset["rows"]:
        row = dict(source)
        failed = (
            row["incentive"] == "opposed"
            and row["surface_kind"] == "assertion"
            and row["mapping_format"] == "prose"
            and row["history"] == "redundant"
        )
        row["action_correct"] = not failed
        row["selected_action"] = row["expected_action"] if not failed else (
            "B" if row["expected_action"] == "A" else "A"
        )
        row["chosen_action_index"] = 0 if row["selected_action"] == "A" else 1
        row["option_reports"] = {
            action: {"correct": True, "choice": "X", "expected": "X"}
            for action in ("A", "B")
        }
        rows.append(row)
    result = analyze_behavior({"metadata": {"run_id": "synthetic"}, "rows": rows}, config)
    assert result["gate_pass"]
    assert result["prose_report_harm_bootstrap_upper"] <= 0.05
    assert result["locked_eligible_by_donor"]["table"]["n"] >= 24
    assert result["discovery_eligible_by_donor"]["opaque"]["n_bases"] >= 18


def test_bfloat16_bit_storage_round_trip() -> None:
    source = np.asarray([1.0, -2.5, 0.125], dtype=np.float32)
    bits = (source.view(np.uint32) >> 16).astype(np.uint16)
    restored = (bits.astype(np.uint32) << 16).view(np.float32)
    assert np.array_equal(source, restored)


def test_downloaded_modal_artifact_path_resolution(tmp_path: Path) -> None:
    remote = "/artifacts/rbg5b/run-123/locked_patches.json.gz"
    local = tmp_path / "modal_artifacts/rbg5b/run-123/locked_patches.json.gz"
    local.parent.mkdir(parents=True)
    local.write_bytes(b"fixture")
    assert _resolve_remote_artifact(remote, tmp_path) == local


def test_geometry_fixture_is_reproducible(tmp_path: Path) -> None:
    config = _config()
    dataset = dataset_payload(config)
    rows = [row for row in dataset["rows"] if row["split"] == "discovery"][:10]
    metadata = [
        {
            "condition_id": row["condition_id"],
            "base_game_id": row["base_game_id"],
            "split": row["split"],
            "frame": row["frame"],
            "incentive": row["incentive"],
            "surface_kind": row["surface_kind"],
            "history": row["history"],
            "mapping_format": row["mapping_format"],
            "action_correct": False,
        }
        for row in rows
    ]
    residual_path = tmp_path / "states.npy"
    array = np.arange(len(rows) * 2 * 5 * 8, dtype=np.float32).reshape(len(rows), 2, 5, 8)
    np.save(residual_path, array.astype(np.float16), allow_pickle=False)
    output = tmp_path / "geometry.json.gz"
    artifact = compute_activation_geometry(
        residual_path,
        metadata,
        ["history_end", "mapping_end", "actions_end", "payoff_end", "answer"],
        [0, 1],
        rows,
        output,
    )
    assert artifact["n_distance_rows"] >= 0
    with gzip.open(output, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["n_cka_rows"] == 6 * 5


def test_full_run_budget_reservation_matches_config() -> None:
    config = _config()
    stage_seconds = {
        "behavior": config["execution"]["estimated_behavior_ceiling_seconds"],
        "discovery": config["execution"]["estimated_capture_ceiling_seconds"],
        "locked": config["execution"]["estimated_patch_ceiling_seconds"],
        "jspace": config["execution"]["estimated_jspace_ceiling_seconds"],
    }
    estimate = sum(
        estimate_cost(
            config["execution"]["gpu"],
            stage_seconds[stage],
            cpu_cores=8 if stage in {"behavior", "jspace"} else 16,
            memory_gib=32 if stage == "behavior" else 64,
        ).buffered_usd
        for stage in stage_seconds
    )
    assert abs(estimate - config["execution"]["stage_reservation_buffered_usd"]) < 0.02
    assert estimate + config["execution"]["prior_v5_buffered_usd_at_freeze"] < config[
        "execution"
    ]["hard_cumulative_v5_cost_limit_usd"]
