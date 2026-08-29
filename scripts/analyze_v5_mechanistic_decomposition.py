from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any

from jspace_policy.mechanistic_decomposition import analyze_locked_patches


def _load(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_probe_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "layer",
        "anchor",
        "target",
        "model_kind",
        "n",
        "balanced_accuracy",
        "bootstrap_95_low",
        "bootstrap_95_high",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/v5/mechanistic_decomposition/experiment.json"),
    )
    parser.add_argument(
        "--behavior",
        type=Path,
        default=Path("results/v5_mechanistic_decomposition/raw/behavior.json"),
    )
    parser.add_argument(
        "--patch-results",
        type=Path,
        default=Path("results/v5_mechanistic_decomposition/raw/locked_patches.json.gz"),
    )
    parser.add_argument(
        "--probe-metrics",
        type=Path,
        default=Path("results/v5_mechanistic_decomposition/raw/locked_probe_metrics.json.gz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/v5_mechanistic_decomposition/analysis"),
    )
    args = parser.parse_args()
    config = _load(args.config)
    behavior = _load(args.behavior)
    patch_analysis = analyze_locked_patches(_load(args.patch_results), config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "locked_patch_analysis.json"
    if output.exists():
        raise SystemExit(f"refusing to overwrite analysis: {output}")
    output.write_text(json.dumps(patch_analysis, indent=2, sort_keys=True) + "\n")
    probe_output = args.output_dir / "locked_probe_metrics.csv"
    if probe_output.exists():
        raise SystemExit(f"refusing to overwrite analysis: {probe_output}")
    _write_probe_csv(_load(args.probe_metrics), probe_output)
    manifest = {
        "schema_version": 1,
        "study_id": "V5-RBG-5",
        "behavior_run_id": behavior["metadata"]["run_id"],
        "dataset_sha256": behavior["metadata"]["dataset_sha256"],
        "behavior_gate_pass": behavior["analysis"]["gate_pass"],
        "locked_patch_analysis": str(output),
        "locked_probe_metrics": str(probe_output),
    }
    manifest_path = args.output_dir / "analysis_manifest.json"
    if manifest_path.exists():
        raise SystemExit(f"refusing to overwrite analysis: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(patch_analysis, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
