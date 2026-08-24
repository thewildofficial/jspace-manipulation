from __future__ import annotations

import json
from pathlib import Path

from jspace_policy.inverse_evidence_games import dataset_payload, verify_dataset_payload

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v5/inverse_evidence/experiment.json"
OUTPUT = ROOT / "configs/v5/inverse_evidence/dataset.json"


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"refusing to overwrite frozen dataset: {OUTPUT}")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload = dataset_payload(config)
    verify_dataset_payload(payload, config)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUTPUT} with {len(payload['rows'])} contexts")


if __name__ == "__main__":
    main()
