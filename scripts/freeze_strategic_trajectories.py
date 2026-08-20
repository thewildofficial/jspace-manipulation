"""Generate and validate the untokenized V2-E2 MVP source corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jspace_policy.strategic_trajectories import dataset_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/v2/strategic_trajectories/experiment.json"
    )
    parser.add_argument(
        "--output", default="configs/v2/strategic_trajectories/dataset_source.json"
    )
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    payload = dataset_payload(config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "rows": len(payload["rows"]),
                "content_sha256": payload["content_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
