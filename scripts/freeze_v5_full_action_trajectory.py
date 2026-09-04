from __future__ import annotations

import json
from pathlib import Path

from jspace_policy.full_action_trajectory import (
    dataset_payload,
    file_sha256,
    verify_dataset_payload,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v5/full_action_trajectory/experiment.json"
MANIFEST = ROOT / "configs/v5/full_action_trajectory/dataset_manifest.json"
OUTPUT = ROOT / "configs/v5/full_action_trajectory/dataset.json"


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"refusing to overwrite frozen dataset: {OUTPUT}")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    source = ROOT / config["source"]["path"]
    actual_source_hash = file_sha256(source)
    if actual_source_hash != config["source"]["file_sha256"]:
        raise SystemExit("source RBG-4 payload hash changed")
    if actual_source_hash != manifest["source_file_sha256"]:
        raise SystemExit("source RBG-4 payload does not match manifest")
    payload = dataset_payload(config, json.loads(source.read_text(encoding="utf-8")))
    verify_dataset_payload(payload, config)
    expected = manifest["expected_content_sha256"]
    if payload["content_sha256"] != expected:
        raise SystemExit("generated dataset does not match frozen manifest")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(payload["content_sha256"])
    print(f"wrote {OUTPUT} with {len(payload['rows'])} trajectories")


if __name__ == "__main__":
    main()
