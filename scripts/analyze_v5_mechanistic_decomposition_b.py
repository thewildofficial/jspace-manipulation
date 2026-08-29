"""Reproducibly analyze the immutable RBG-5B outputs offline.

This command intentionally does not import Modal or contact the control plane.
Remote manifests retain their absolute Modal-volume paths; the analysis module
resolves those paths under ``results/v5_mechanistic_decomposition_b/modal_artifacts``
when the workflow has downloaded the corresponding volume trees.
"""

from __future__ import annotations

import json
from pathlib import Path

from jspace_policy.mechanistic_decomposition_analysis import analyze_study
from jspace_policy.mechanistic_decomposition_games import verify_dataset_payload

CONFIG_PATH = Path("configs/v5/mechanistic_decomposition_b/experiment.json")
DATASET_PATH = Path("configs/v5/mechanistic_decomposition_b/dataset.json")
MANIFEST_PATH = Path("configs/v5/mechanistic_decomposition_b/dataset_manifest.json")
RESULT_ROOT = Path("results/v5_mechanistic_decomposition_b")


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    verify_dataset_payload(dataset, config)
    if dataset["content_sha256"] != manifest["expected_content_sha256"]:
        raise SystemExit("dataset does not match the committed RBG-5B manifest")
    if config["dataset"]["expected_content_sha256"] != manifest["expected_content_sha256"]:
        raise SystemExit("config does not match the committed RBG-5B manifest")
    output_path = RESULT_ROOT / "analysis/final_analysis.json"
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite analysis: {output_path}")
    output = analyze_study(config, dataset, RESULT_ROOT)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
