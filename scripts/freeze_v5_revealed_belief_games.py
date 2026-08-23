from __future__ import annotations

import argparse
import json
from pathlib import Path

from jspace_policy.revealed_belief_games import dataset_payload, verify_dataset_payload

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/v5/revealed_belief_games/experiment.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "configs/v5/revealed_belief_games/dataset.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite frozen dataset: {args.output}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    payload = dataset_payload(config)
    verify_dataset_payload(payload, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output} with {len(payload['rows'])} contexts")


if __name__ == "__main__":
    main()
