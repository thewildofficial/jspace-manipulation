"""RBG-5B entrypoints using the shared, tested V5 mechanistic runner.

The local paths and artifact namespace are separate from RBG-5.  Remote stages
receive the immutable config and dataset as values, so they cannot accidentally
read or overwrite the earlier study.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import modal

import modal_v5_mechanistic_decomposition as base

CONFIG_PATH = Path("configs/v5/mechanistic_decomposition_b/experiment.json")
DATASET_PATH = Path("configs/v5/mechanistic_decomposition_b/dataset.json")
MANIFEST_PATH = Path("configs/v5/mechanistic_decomposition_b/dataset_manifest.json")
RESULT_ROOT = Path("results/v5_mechanistic_decomposition_b")

app: modal.App = base.app


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_new(path: Path, value: object) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite prospective artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _config() -> dict:
    return _load_json(CONFIG_PATH)


def _dataset(config: dict) -> dict:
    dataset = _load_json(DATASET_PATH)
    base._validate(config, dataset)
    manifest = _load_json(MANIFEST_PATH)
    if dataset["content_sha256"] != manifest["expected_content_sha256"]:
        raise RuntimeError("dataset does not match the committed RBG-5B manifest")
    if dataset["config_sha256"] != manifest["expected_config_sha256"]:
        raise RuntimeError("dataset/config hash does not match the committed RBG-5B manifest")
    return dataset


def _admit_full_run(config: dict) -> None:
    current = base._v5_buffered_total()
    estimates = sum(
        base._stage_estimate(config, stage).buffered_usd
        for stage in ("behavior", "discovery", "locked", "jspace")
    )
    limit = float(config["execution"]["hard_cumulative_v5_cost_limit_usd"])
    if current + estimates > limit:
        raise RuntimeError(
            f"RBG-5B reservation refused: ${current + estimates:.3f} would exceed ${limit:.2f}"
        )
    declared = float(config["execution"]["stage_reservation_buffered_usd"])
    if abs(estimates - declared) > 0.02:
        raise RuntimeError("RBG-5B stage reservation does not match executable ceilings")


@app.local_entrypoint(name="rbg5b_freeze_dataset")
def freeze_dataset() -> None:
    from jspace_policy.mechanistic_decomposition_games import (
        dataset_payload,
        verify_dataset_payload,
    )

    config = _config()
    payload = dataset_payload(config)
    verify_dataset_payload(payload, config)
    manifest = _load_json(MANIFEST_PATH)
    if payload["content_sha256"] != manifest["expected_content_sha256"]:
        raise RuntimeError("generated dataset differs from the frozen RBG-5B manifest")
    _write_new(DATASET_PATH, payload)
    print(json.dumps({"rows": len(payload["rows"]), "sha256": payload["content_sha256"]}))


@app.local_entrypoint(name="rbg5b_preflight")
def preflight() -> None:
    config = _config()
    dataset = _dataset(config)
    payload = json.loads(base.preflight_remote.remote(dataset, config))
    _write_new(RESULT_ROOT / "raw/preflight.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


@app.local_entrypoint(name="rbg5b_behavior")
def behavior() -> None:
    if not base._tracked_tree_clean():
        raise RuntimeError("behavior execution requires a clean tracked worktree")
    config = _config()
    dataset = _dataset(config)
    _admit_full_run(config)
    reservation = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "created_at": datetime.now(UTC).isoformat(),
        "prior_v5_buffered_usd": base._v5_buffered_total(),
        "stage_buffered_usd": {
            stage: base._stage_estimate(config, stage).buffered_usd
            for stage in ("behavior", "discovery", "locked", "jspace")
        },
        "hard_limit_usd": config["execution"]["hard_cumulative_v5_cost_limit_usd"],
    }
    _write_new(RESULT_ROOT / "raw/cost_reservation.json", reservation)
    payload = json.loads(base.behavior_remote.remote(dataset, config, base._git_head()))
    _write_new(RESULT_ROOT / "raw/behavior.json", payload)
    _write_new(RESULT_ROOT / "analysis/behavior_analysis.json", payload["analysis"])
    base._record_stage(config, payload, "behavior")
    print(json.dumps(payload["analysis"], indent=2, sort_keys=True))


@app.local_entrypoint(name="rbg5b_discovery")
def discovery() -> None:
    if not base._tracked_tree_clean():
        raise RuntimeError("discovery execution requires a clean tracked worktree")
    config = _config()
    dataset = _dataset(config)
    behavior_payload = _load_json(RESULT_ROOT / "raw/behavior.json")
    base._admit_stage(config, "discovery")
    payload = json.loads(
        base.discovery_remote.remote(dataset, behavior_payload, config, base._git_head())
    )
    _write_new(RESULT_ROOT / "raw/discovery_manifest.json", payload)
    _write_new(RESULT_ROOT / "probe_freeze.json", payload["probe_freeze"])
    _write_new(RESULT_ROOT / "patch_freeze.json", payload["patch_freeze"])
    base._record_stage(config, payload, "discovery")
    print(json.dumps(payload["patch_freeze"], indent=2, sort_keys=True))


@app.local_entrypoint(name="rbg5b_locked")
def locked() -> None:
    if not base._tracked_tree_clean():
        raise RuntimeError("locked execution requires a clean tracked worktree")
    config = _config()
    dataset = _dataset(config)
    behavior_payload = _load_json(RESULT_ROOT / "raw/behavior.json")
    discovery_manifest = _load_json(RESULT_ROOT / "raw/discovery_manifest.json")
    base._admit_stage(config, "locked")
    payload = json.loads(
        base.locked_remote.remote(
            dataset, behavior_payload, discovery_manifest, config, base._git_head()
        )
    )
    _write_new(RESULT_ROOT / "raw/locked_manifest.json", payload)
    base._record_stage(config, payload, "locked")
    print(json.dumps(payload["counts"], indent=2, sort_keys=True))


@app.local_entrypoint(name="rbg5b_jspace")
def jspace() -> None:
    config = _config()
    dataset = _dataset(config)
    discovery_manifest = _load_json(RESULT_ROOT / "raw/discovery_manifest.json")
    locked_path = RESULT_ROOT / "raw/locked_manifest.json"
    if not locked_path.exists():
        if discovery_manifest["patch_freeze"]["selected"] is None:
            payload = {
                "schema_version": 1,
                "study_id": config["study_id"],
                "status": "discovery_only_no_positive_patch",
                "created_at": datetime.now(UTC).isoformat(),
            }
            _write_new(RESULT_ROOT / "raw/jspace_manifest.json", payload)
            print(json.dumps(payload, indent=2, sort_keys=True))
            return
        raise RuntimeError("locked manifest is required before RBG-5B J-space")
    locked_manifest = _load_json(locked_path)
    base._admit_stage(config, "jspace")
    payload = json.loads(
        base.jspace_remote.remote(
            dataset, discovery_manifest, locked_manifest, config, base._git_head()
        )
    )
    _write_new(RESULT_ROOT / "raw/jspace_manifest.json", payload)
    base._record_stage(config, payload, "jspace")
    print(json.dumps(payload["artifact"], indent=2, sort_keys=True))


@app.local_entrypoint(name="rbg5b_analyze")
def analyze() -> None:
    from jspace_policy.mechanistic_decomposition_analysis import analyze_study

    config = _config()
    dataset = _dataset(config)
    output = analyze_study(config, dataset, RESULT_ROOT)
    _write_new(RESULT_ROOT / "analysis/final_analysis.json", output)
    print(json.dumps(output, indent=2, sort_keys=True))
